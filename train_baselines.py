#!/usr/bin/env python
"""Run official/reputable external SNN-HSI model code on our clean splits.

This is intentionally separate from `train.py`: external paper
code often carries hard-coded assumptions, so each adapter must record its
source repo, commit, and any compatibility fix in the result JSON.
"""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from spikesst.config import Config
from spikesst.data import build_dataloaders
from spikesst.energy import ann_gflops, count_params, profile_energy
from spikesst.engine import compute_metrics, evaluate, train_model


BASELINES = {
    'brain_inspired_snn': {
        'display': 'Brain-inspired SNN',
        'repo': 'https://github.com/Katherine-Cao/HSI_SNN',
        'path': 'baselines/HSI_SNN',
        'module': 'models/TGRS.py',
        'class': 'TGRS',
        'timesteps': 30,
        'leak_mem': 0.7,
        'adapter': 'official_model_class',
    },
    'frm_snn': {
        'display': 'FRM-SNN',
        'repo': 'https://github.com/Li-SNN/FRM-SNN',
        'path': 'baselines/FRM-SNN',
        'module': 'SNN_SWRB.py',
        'class': 'TGRS',
        'timesteps': 40,
        'leak_mem': 0.7,
        'adapter': 'official_model_file_with_classifier_head_check',
    },
}


def _load_class(path, class_name):
    spec = importlib.util.spec_from_file_location('external_snn_model', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


def _git_commit(path):
    try:
        return subprocess.check_output(
            ['git', '-C', path, 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ''


def _config_hash(d):
    blob = json.dumps(d, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def build_external_model(baseline_id, in_channels, num_classes, cfg):
    meta = BASELINES[baseline_id]
    repo_path = os.path.abspath(meta['path'])
    model_path = os.path.join(repo_path, meta['module'])
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing external model file {model_path}. Clone {meta['repo']} "
            f"into {meta['path']} first.")
    cls = _load_class(model_path, meta['class'])
    model = cls(meta['timesteps'], leak_mem=meta['leak_mem'],
                img_size=cfg.window_size, num_cls=num_classes,
                input_dim=in_channels)
    adapter_notes = []
    if baseline_id == 'frm_snn':
        pooled = cfg.window_size // 4
        expected = 256 * pooled * pooled
        if getattr(model.fc1, 'in_features', expected) != expected:
            old = model.fc1.in_features
            model.fc1 = nn.Linear(expected, num_classes, bias=False)
            nn.init.xavier_uniform_(model.fc1.weight, gain=5)
            adapter_notes.append(
                f'replaced classifier head in_features {old} -> {expected} '
                f'for window_size={cfg.window_size}')
    return model, adapter_notes


def _arch_config(baseline_id, cfg, adapter_notes):
    meta = BASELINES[baseline_id]
    return {
        'external_baseline': baseline_id,
        'display_name': meta['display'],
        'source_repo': meta['repo'],
        'source_commit': _git_commit(meta['path']),
        'source_module': meta['module'],
        'adapter': meta['adapter'],
        'adapter_notes': adapter_notes,
        'timesteps': meta['timesteps'],
        'leak_mem': meta['leak_mem'],
        'window_size': cfg.window_size,
        'n_components': cfg.n_components,
        'epochs': cfg.epochs,
        'optimizer_recipe': 'snn_hsi_shared_adamw',
    }


def run_one(baseline_id, dataset, split, seed, cfg, device, out_dir):
    cfg = dataclasses.replace(cfg, model=baseline_id, dataset=dataset,
                              split=split, seed=seed, output_dir=out_dir)
    tag = f'{dataset}_{baseline_id}_{split}_s{seed}'
    if cfg.augment:
        tag += '_aug'
    out_json = os.path.join(out_dir, f'{tag}.json')
    if os.path.exists(out_json):
        try:
            with open(out_json) as f:
                done = json.load(f)
            print(f'SKIP (done): {tag}')
            return done
        except json.JSONDecodeError:
            print(f'RERUN (corrupt JSON): {tag}')

    print(f'\n=== {tag} ===', flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, val_loader, test_loader, data_meta = build_dataloaders(cfg)
    model, adapter_notes = build_external_model(
        baseline_id, cfg.n_components, data_meta['num_classes'], cfg)
    model = model.to(device)
    params = count_params(model)
    print(f'  Params: {params:,}')

    model, history, train_time = train_model(model, train_loader, val_loader,
                                             cfg, device)
    start = time.time()
    _, _, y_pred, y_true = evaluate(model, test_loader, nn.CrossEntropyLoss(),
                                    device)
    test_time = time.time() - start
    metrics = compute_metrics(y_true, y_pred, data_meta['target_names'])

    gflops = ann_gflops(model, (cfg.n_components, cfg.window_size,
                                cfg.window_size), device)
    energy = profile_energy(model, test_loader, device, cfg,
                            dense_gflops=gflops)

    arch = _arch_config(baseline_id, cfg, adapter_notes)
    result = {
        'tag': tag,
        'model': baseline_id,
        'external_baseline': True,
        'dataset': dataset,
        'split': split,
        'seed': seed,
        'attn_variant': None,
        'OA': metrics['OA'],
        'AA': metrics['AA'],
        'Kappa': metrics['Kappa'],
        'Macro_F1': metrics['Macro_F1'],
        'per_class': metrics['per_class'].tolist(),
        'params': params,
        'ANN_GFLOPs': gflops,
        'train_time': train_time,
        'test_time': test_time,
        'n_train': data_meta['n_train'],
        'n_val': data_meta['n_val'],
        'n_test': data_meta['n_test'],
        'timesteps': BASELINES[baseline_id]['timesteps'],
        'window_size': cfg.window_size,
        'n_components': cfg.n_components,
        'train_ratio': cfg.train_ratio,
        'use_sta': None,
        'use_sce': None,
        'augment': cfg.augment,
        'arch_variant': 'official_external_adapter',
        'arch_config': arch,
        'arch_config_hash': _config_hash(arch),
        'E_SNN_uJ': energy['E_SNN_uJ'],
        'E_ANN_uJ': energy['E_ANN_uJ'],
        'energy_ratio': energy['energy_ratio'],
        'SOPs_M': energy['SOPs_M'],
        'MAC_ops_M': energy['MAC_ops_M'],
        'mean_firing_rate': energy['mean_firing_rate'],
        'n_profile_samples': energy['n_profile_samples'],
        'lif_rates': energy['lif_rates'],
        'per_layer': energy['per_layer'],
        'components': energy['components'],
    }
    if data_meta.get('split_manifest'):
        split_meta = data_meta['split_manifest']
        result.update({
            'split_hash': split_meta.get('sha256_16'),
            'split_manifest': split_meta,
        })

    ckpt_dir = os.path.join(out_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{tag}.pt')
    torch.save(model.state_dict(), ckpt_path)
    result['checkpoint'] = os.path.relpath(ckpt_path, out_dir)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} "
          f"E_ratio={energy['energy_ratio']:.4f} "
          f"fr={energy['mean_firing_rate']:.3f}", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baselines', nargs='+', choices=sorted(BASELINES),
                   default=['brain_inspired_snn'])
    p.add_argument('--datasets', nargs='+', default=['PU'])
    p.add_argument('--splits', nargs='+', default=['disjoint'])
    p.add_argument('--seeds', nargs='+', type=int, default=[345])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--train-ratio', type=float, default=0.05)
    p.add_argument('--out', default='results/baselines')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--window-size', type=int, default=13)
    p.add_argument('--n-components', type=int, default=20)
    p.add_argument('--split-manifest', default='')
    p.add_argument('--no-write-split-manifest', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise SystemExit('External official SNN code hard-codes .cuda(); run on a GPU node.')
    os.makedirs(args.out, exist_ok=True)
    cfg = Config(epochs=args.epochs, train_ratio=args.train_ratio,
                 augment=args.augment, patience=args.patience,
                 window_size=args.window_size, n_components=args.n_components,
                 split_manifest=args.split_manifest,
                 write_split_manifest=not args.no_write_split_manifest)
    results = []
    for dataset in args.datasets:
        for split in args.splits:
            for baseline_id in args.baselines:
                for seed in args.seeds:
                    results.append(run_one(baseline_id, dataset, split, seed,
                                           cfg, device, args.out))
    return results


if __name__ == '__main__':
    main()
