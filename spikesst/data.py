"""Data pipeline: loading, PCA (train-only fit), random & spatially disjoint splits,
memory-lazy patch datasets."""
import hashlib
import contextlib
import fcntl
import json
import os
import numpy as np
import scipy.io as sio
import torch
from scipy.ndimage import maximum_filter
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

from .config import DATASET_INFO


def load_hsi(dataset_name, data_path):
    info = DATASET_INFO[dataset_name]
    hsi = sio.loadmat(os.path.join(data_path, info['data_file']))[info['data_key']]
    gt = sio.loadmat(os.path.join(data_path, info['gt_file']))[info['gt_key']]
    return hsi.astype(np.float32), gt.astype(np.int64), info


def random_split_coords(gt, train_ratio, val_ratio, seed):
    """Stratified random pixel split. Returns (train, val, test) arrays of (row, col)."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(gt[gt > 0]):
        coords = np.argwhere(gt == c)
        rng.shuffle(coords)
        n = len(coords)
        n_tr = max(int(round(train_ratio * n)), 2)
        n_va = max(int(round(val_ratio * n)), 1)
        train.append(coords[:n_tr])
        val.append(coords[n_tr:n_tr + n_va])
        test.append(coords[n_tr + n_va:])
    return np.concatenate(train), np.concatenate(val), np.concatenate(test)


def split_digest(train, val, test):
    """Short SHA-256 digest of the exact coordinate arrays."""
    h = hashlib.sha256()
    for arr in (train, val, test):
        h.update(np.ascontiguousarray(arr, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


class SplitManifestError(RuntimeError):
    """A requested split manifest entry is missing, malformed, or hash-mismatched.

    Raised as its own type so callers can abort the whole job instead of skipping
    the cell: a silently missing cell disappears from the results table, which is
    exactly the failure mode we must never have when methods are required to share
    identical splits.
    """


def _manifest_entries(path):
    if not os.path.exists(path):
        raise SplitManifestError(f'Split manifest not found: {path}')
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and 'splits' in data:
        return data['splits']
    if isinstance(data, list):
        return data
    raise SplitManifestError(f'Unsupported split manifest format: {path}')


def find_split_manifest_entry(path, dataset, split, seed, window_size):
    """Return the manifest entry matching an experiment split."""
    for entry in _manifest_entries(path):
        if (entry.get('dataset') == dataset and entry.get('split') == split and
                int(entry.get('seed')) == int(seed) and
                int(entry.get('window_size')) == int(window_size)):
            return entry
    raise SplitManifestError(
        f'No split manifest entry for {dataset}/{split}/s{seed}/w{window_size} in {path}')


def load_split_from_manifest(path, dataset, split, seed, window_size):
    """Load train/val/test coordinates from a manifest entry and verify its hash."""
    entry = find_split_manifest_entry(path, dataset, split, seed, window_size)
    split_file = entry['file']
    if not os.path.isabs(split_file):
        split_file = os.path.join(os.path.dirname(path), split_file)
    data = np.load(split_file)
    tr = np.asarray(data['train'], dtype=np.int32)
    va = np.asarray(data['val'], dtype=np.int32)
    te = np.asarray(data['test'], dtype=np.int32)
    digest = split_digest(tr, va, te)
    expected = entry.get('sha256_16') or entry.get('split_hash')
    if expected and digest != expected:
        raise SplitManifestError(
            f'Split hash mismatch for {split_file}: {digest} != {expected}')
    return tr, va, te, dict(entry, sha256_16=digest, file=os.path.basename(split_file))


def write_split_manifest_entry(out_dir, dataset, split, seed, window_size,
                               train_ratio, val_ratio, block, train, val, test,
                               num_classes):
    """Persist exact coordinates and merge/update the local manifest."""
    split_dir = os.path.join(out_dir, 'splits')
    os.makedirs(split_dir, exist_ok=True)
    name = f'{dataset}_{split}_s{seed}_w{window_size}.npz'
    np.savez_compressed(os.path.join(split_dir, name),
                        train=np.asarray(train, dtype=np.int32),
                        val=np.asarray(val, dtype=np.int32),
                        test=np.asarray(test, dtype=np.int32))
    entry = {
        'dataset': dataset, 'split': split, 'seed': int(seed),
        'window_size': int(window_size), 'train_ratio': float(train_ratio),
        'val_ratio': float(val_ratio), 'block': int(block),
        'n_train': int(len(train)), 'n_val': int(len(val)), 'n_test': int(len(test)),
        'num_classes': int(num_classes), 'sha256_16': split_digest(train, val, test),
        'file': name,
    }
    manifest_path = os.path.join(split_dir, 'manifest.json')
    lock_path = manifest_path + '.lock'
    with open(lock_path, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = []
        if os.path.exists(manifest_path) and os.path.getsize(manifest_path) > 0:
            with open(manifest_path) as f:
                with contextlib.suppress(json.JSONDecodeError):
                    old = json.load(f)
                    manifest = old['splits'] if isinstance(old, dict) and 'splits' in old else old
        key = (dataset, split, int(seed), int(window_size))
        manifest = [e for e in manifest
                    if (e.get('dataset'), e.get('split'), int(e.get('seed')),
                        int(e.get('window_size'))) != key]
        manifest.append(entry)
        manifest.sort(key=lambda e: (e['dataset'], e['split'], e['seed'], e['window_size']))
        tmp_path = manifest_path + f'.tmp.{os.getpid()}'
        with open(tmp_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)
    return dict(entry, manifest_path=manifest_path)


def disjoint_split_coords(gt, train_ratio, val_ratio, seed, block=32, margin=4):
    """Spatially disjoint split.

    The scene is tiled into `block`x`block` blocks. Blocks are greedily assigned to a
    train/val pool until every class reaches its target pixel count; all remaining
    blocks form the test pool. Train/val pixels are subsampled from the pool blocks to
    the target counts, and test pixels are then dropped if their patch would overlap
    any selected train/val pixel's patch.

    `margin` is the patch radius m (= window_size // 2). Two patches of radius m
    overlap whenever their centers lie within Chebyshev distance 2m, so the exclusion
    zone is dilated by 2m -- not m. Dilating by only m (the behaviour before this fix)
    left ~15% of test patches overlapping a training patch on Pavia University and
    WHU-Hi-HanChuan.
    """
    rng = np.random.default_rng(seed)
    h, w = gt.shape
    classes = np.unique(gt[gt > 0])
    counts = {c: int((gt == c).sum()) for c in classes}
    target = {c: max(int(round((train_ratio + val_ratio) * counts[c])), 3) for c in classes}
    deficit = dict(target)

    block_ids = np.stack([np.repeat(np.arange(h) // block, w).reshape(h, w),
                          np.tile(np.arange(w) // block, h).reshape(h, w)], axis=-1)
    flat_bid = block_ids[..., 0] * ((w // block) + 1) + block_ids[..., 1]
    unique_blocks = np.unique(flat_bid[gt > 0])
    rng.shuffle(unique_blocks)

    block_classes = {b: np.unique(gt[(flat_bid == b) & (gt > 0)]) for b in unique_blocks}
    test_blocks_left = {c: sum(1 for b in unique_blocks if c in block_classes[b])
                        for c in classes}

    trainval_blocks = set()
    # pass 1 with a guard: never take the last remaining test block of any class;
    # pass 2 lifts the guard for classes whose quota is still unmet (warned below)
    for guard in (True, False):
        for b in unique_blocks:
            if b in trainval_blocks:
                continue
            cls_in_block = block_classes[b]
            if not any(deficit[c] > 0 for c in cls_in_block):
                continue
            if guard and any(test_blocks_left[c] <= 1 for c in cls_in_block):
                continue
            trainval_blocks.add(b)
            mask = (flat_bid == b) & (gt > 0)
            cls, n_in_block = np.unique(gt[mask], return_counts=True)
            for c, n in zip(cls, n_in_block):
                deficit[c] = max(deficit[c] - int(n), 0)
                test_blocks_left[c] -= 1

    in_pool = np.isin(flat_bid, list(trainval_blocks)) & (gt > 0)
    train, val = [], []
    selected_mask = np.zeros_like(gt, dtype=bool)
    for c in classes:
        coords = np.argwhere(in_pool & (gt == c))
        rng.shuffle(coords)
        n_take = min(target[c], len(coords))
        take = coords[:n_take]
        n_tr = max(int(round(n_take * train_ratio / (train_ratio + val_ratio))), 1)
        train.append(take[:n_tr])
        val.append(take[n_tr:] if n_take > n_tr else take[:1])
        selected_mask[take[:, 0], take[:, 1]] = True

    # Exclude test pixels whose patch would overlap any train/val pixel's patch.
    # Patches of radius `margin` overlap up to Chebyshev distance 2*margin.
    buffer = maximum_filter(selected_mask.astype(np.uint8),
                            size=2 * (2 * margin) + 1) > 0
    test_ok = (gt > 0) & (~in_pool) & (~buffer)
    test = np.argwhere(test_ok)

    missing = [int(c) for c in classes if not (gt[test[:, 0], test[:, 1]] == c).any()]
    if missing:
        print(f'  [disjoint split] WARNING: classes with no test pixels: {missing}')
    return np.concatenate(train), np.concatenate(val), test


def pca_train_only(hsi, train_coords, n_components):
    """Fit PCA on training-pixel spectra only, transform the full cube."""
    h, w, bands = hsi.shape
    train_spectra = hsi[train_coords[:, 0], train_coords[:, 1], :]
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(train_spectra)
    reduced = pca.transform(hsi.reshape(-1, bands)).astype(np.float32)
    return reduced.reshape(h, w, n_components)


class PatchDataset(Dataset):
    """Lazily slices patches out of a single padded cube (no NxWxWxK materialization)."""

    def __init__(self, padded_cube, coords, labels, window_size, augment=False):
        # padded_cube: torch tensor (K, H+2m, W+2m), margin m = window_size // 2
        self.cube = padded_cube
        self.coords = coords
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.ws = window_size
        self.augment = augment

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c = self.coords[idx]
        patch = self.cube[:, r:r + self.ws, c:c + self.ws]
        if self.augment:
            k = int(torch.randint(0, 4, (1,)))
            if k:
                patch = torch.rot90(patch, k, dims=(1, 2))
            if torch.rand(1) < 0.5:
                patch = torch.flip(patch, dims=(2,))
            patch = patch.contiguous()
        return patch, self.labels[idx]


def pad_cube(hsi_pca, window_size):
    margin = window_size // 2
    padded = np.pad(hsi_pca, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')
    return torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))


def build_dataloaders(cfg):
    """Full pipeline. Returns train/val/test loaders + metadata dict."""
    hsi, gt, info = load_hsi(cfg.dataset, cfg.data_path)
    margin = cfg.window_size // 2

    split_entry = None
    if getattr(cfg, 'split_manifest', ''):
        tr, va, te, split_entry = load_split_from_manifest(
            cfg.split_manifest, cfg.dataset, cfg.split, cfg.seed, cfg.window_size)
    elif cfg.split == 'random':
        tr, va, te = random_split_coords(gt, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    elif cfg.split == 'disjoint':
        tr, va, te = disjoint_split_coords(gt, cfg.train_ratio, cfg.val_ratio, cfg.seed,
                                           block=cfg.disjoint_block, margin=margin)
    else:
        raise ValueError(f'Unknown split: {cfg.split}')
    if getattr(cfg, 'write_split_manifest', False) and (
            split_entry is None or
            os.path.dirname(getattr(cfg, 'split_manifest', '')) != os.path.join(cfg.output_dir, 'splits')):
        split_entry = write_split_manifest_entry(
            cfg.output_dir, cfg.dataset, cfg.split, cfg.seed, cfg.window_size,
            cfg.train_ratio, cfg.val_ratio, cfg.disjoint_block, tr, va, te,
            info['num_classes'])

    hsi_pca = pca_train_only(hsi, tr, cfg.n_components)
    cube = pad_cube(hsi_pca, cfg.window_size)

    def labels_of(coords):
        return gt[coords[:, 0], coords[:, 1]] - 1

    ds_tr = PatchDataset(cube, tr, labels_of(tr), cfg.window_size,
                         augment=getattr(cfg, 'augment', False))
    ds_va = PatchDataset(cube, va, labels_of(va), cfg.window_size)
    ds_te = PatchDataset(cube, te, labels_of(te), cfg.window_size)

    train_loader = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    # eval_batch_size lets profiling jobs cap memory: large patches (w=13) over
    # 300k+ test pixels can exhaust the GPU at the default 256.
    eval_bs = getattr(cfg, 'eval_batch_size', 0) or max(cfg.batch_size, 256)
    val_loader = DataLoader(ds_va, batch_size=eval_bs, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(ds_te, batch_size=eval_bs, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    meta = {
        'num_classes': info['num_classes'], 'target_names': info['target_names'],
        'gt': gt, 'cube': cube, 'hsi_pca': hsi_pca,
        'n_train': len(tr), 'n_val': len(va), 'n_test': len(te),
        'split_manifest': split_entry,
    }
    print(f'  Data: {cfg.dataset} {hsi.shape} split={cfg.split} '
          f'train={len(tr)} val={len(va)} test={len(te)}')
    return train_loader, val_loader, test_loader, meta


class AllPixelDataset(Dataset):
    """Every pixel of the scene, for full-scene segmentation maps."""

    def __init__(self, padded_cube, height, width, window_size):
        self.cube = padded_cube
        self.h, self.w = height, width
        self.ws = window_size

    def __len__(self):
        return self.h * self.w

    def __getitem__(self, idx):
        r, c = divmod(idx, self.w)
        return self.cube[:, r:r + self.ws, c:c + self.ws]
