#!/usr/bin/env python
"""Re-profile saved checkpoints with the corrected energy accounting (v2).

Forward-only: no retraining. Results are written to a SEPARATE root
(`results/energy/` by default) and never overwrite the energy
fields of the original run JSONs, so the earlier numbers stay auditable. Each
record is keyed by `result_path`, `checkpoint`, `energy_accounting_version` and
`protocol_hash`.

Needs a GPU for the external baselines: their official code hard-codes `.cuda()`
on every membrane tensor.

Usage:
  python profile_energy.py --src results/baselines
  python profile_energy.py --src results/spikesst
"""
import argparse
import dataclasses
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from spikesst.config import Config, DATASET_INFO
from spikesst.data import build_dataloaders
from spikesst.energy import (ENERGY_ACCOUNTING_VERSION, ann_gflops,
                            profile_energy)
from spikesst.models import build_model


def protocol_hash(row):
    """Identifies the evaluation protocol a profile belongs to."""
    key = {k: row.get(k) for k in
           ('dataset', 'split', 'seed', 'window_size', 'n_components',
            'train_ratio', 'split_hash')}
    blob = json.dumps(key, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_model(row, cfg, num_classes, device):
    """Rebuild the architecture this run used, ours or external."""
    model_name = row['model']
    if model_name == 'SpikeSST':
        cfg = dataclasses.replace(
            cfg, attn_variant=row.get('attn_variant') or 'sdsa',
            arch_variant=row.get('arch_variant') or 'base')
        model = build_model(model_name, cfg.n_components, num_classes, cfg)
        notes = []
    else:
        from train_baselines import build_external_model
        model, notes = build_external_model(model_name, cfg.n_components,
                                            num_classes, cfg)
    return model.to(device), notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='directory of run JSONs')
    p.add_argument('--out', default='results/energy')
    p.add_argument('--eval-batch-size', type=int, default=128,
                   help='cap profiling memory; w=13 over 300k+ patches is heavy')
    p.add_argument('--max-batches', type=int, default=0,
                   help='0 = whole test set (default)')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('WARNING: no CUDA. External baselines hard-code .cuda() and will fail.')

    files = sorted(glob.glob(os.path.join(args.src, '*.json')))
    print(f'{len(files)} runs in {args.src}; profiling to {args.out}/ on {device}')
    done = failed = 0

    for f in files:
        with open(f) as fh:
            row = json.load(fh)
        tag = row.get('tag') or os.path.splitext(os.path.basename(f))[0]
        out_json = os.path.join(args.out, f'{tag}.json')
        if os.path.exists(out_json) and not args.overwrite:
            print(f'  skip (done): {tag}')
            continue

        ckpt = row.get('checkpoint')
        ckpt_path = ckpt if ckpt and os.path.isabs(ckpt) else \
            os.path.join(os.path.dirname(f), ckpt or '')
        if not ckpt or not os.path.exists(ckpt_path):
            print(f'  SKIP (no checkpoint): {tag}')
            failed += 1
            continue

        try:
            cfg = dataclasses.replace(
                Config(), dataset=row['dataset'], split=row['split'],
                seed=row['seed'], window_size=row['window_size'],
                n_components=row['n_components'],
                train_ratio=row.get('train_ratio', 0.05),
                timesteps=row.get('timesteps') or Config().timesteps,
                augment=False, eval_batch_size=args.eval_batch_size,
                split_manifest=(row.get('split_manifest') or {}).get('manifest_path', ''),
                write_split_manifest=False)
            _, _, test_loader, meta = build_dataloaders(cfg)

            # the split must be the one this run was scored on
            got = (meta.get('split_manifest') or {}).get('sha256_16')
            if row.get('split_hash') and got and got != row['split_hash']:
                raise RuntimeError(
                    f'split hash mismatch: run={row["split_hash"]} rebuilt={got}')

            model, notes = load_model(row, cfg, meta['num_classes'], device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            gflops = ann_gflops(model, (cfg.n_components, cfg.window_size,
                                        cfg.window_size), device)
            energy = profile_energy(
                model, test_loader, device, cfg, dense_gflops=gflops,
                max_batches=args.max_batches or None)

            rec = {
                'tag': tag, 'model': row['model'], 'dataset': row['dataset'],
                # needed to rebuild the method key ('SpikeSST'), which is how
                # the reporting layer joins profiles to runs
                'attn_variant': row.get('attn_variant'),
                'arch_variant': row.get('arch_variant', 'base'),
                'split': row['split'], 'seed': row['seed'],
                'window_size': row['window_size'],
                'n_components': row['n_components'],
                'timesteps': row.get('timesteps'),
                'split_hash': row.get('split_hash'),
                'protocol_hash': protocol_hash(row),
                'source_result': os.path.abspath(f),
                'checkpoint': os.path.abspath(ckpt_path),
                'ANN_GFLOPs': gflops, 'adapter_notes': notes,
            }
            rec.update(energy)
            with open(out_json, 'w') as fh:
                json.dump(rec, fh, indent=2)

            print(f'  {tag}: primary={energy["E_primary_uJ"]:9.2f} uJ  '
                  f'optimistic={energy["E_optimistic_lower_uJ"]:9.2f} uJ  '
                  f'(AC {energy["binary_ac_uJ"]:.1f} / multibit-MAC '
                  f'{energy["multibit_mac_uJ"]:.1f} / static {energy["static_input_mac_uJ"]:.1f})  '
                  f'fr={energy["mean_firing_rate"]:.3f}')
            done += 1
        except Exception as e:
            print(f'  FAILED {tag}: {type(e).__name__}: {e}')
            failed += 1

    print(f'\nprofiled {done}, failed/skipped {failed}')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
