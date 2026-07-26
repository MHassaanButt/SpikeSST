#!/usr/bin/env python
"""Run (model x dataset x split x seeds) experiments with per-experiment JSON
checkpointing (safe to re-run; finished combos are skipped).

Example:
  python train.py --models SpikeSST --datasets PU HC \
      --splits random --seeds 345 445 545 --attn-variant sdsa
"""
import argparse
import dataclasses
import json
import hashlib
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from spikesst.config import Config
from spikesst.data import SplitManifestError, build_dataloaders
from spikesst.energy import ann_gflops, count_params, profile_energy
from spikesst.engine import compute_metrics, evaluate, extract_features, train_model
from spikesst.models import build_model

SNN_MODELS = {'SpikeSST'}


def _arch_config(cfg, model_name):
    fields = ['model', 'attn_variant', 'arch_variant', 'embed_dim', 'num_heads',
              'num_layers', 'timesteps', 'window_size', 'n_components', 'dropout',
              'use_sta', 'use_sce', 'spectral_stem', 'plif']
    return {k: getattr(cfg, k) for k in fields if hasattr(cfg, k)}


def _config_hash(d):
    blob = json.dumps(d, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_one(model_name, dataset, split, seed, cfg, device, out_dir, plots=False,
            skip_energy=False):
    cfg = dataclasses.replace(cfg, model=model_name, dataset=dataset, split=split,
                              seed=seed, output_dir=out_dir)
    tag = f'{dataset}_{model_name}_{split}_s{seed}'
    if model_name == 'SpikeSST':
        tag = f'{dataset}_{model_name}-{cfg.attn_variant}_{split}_s{seed}'
    if cfg.augment:
        tag += '_aug'
    if getattr(cfg, 'arch_variant', 'base') != 'base':
        tag += f'_{cfg.arch_variant}'
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

    train_loader, val_loader, test_loader, meta = build_dataloaders(cfg)
    model = build_model(model_name, cfg.n_components, meta['num_classes'], cfg).to(device)
    params = count_params(model)
    print(f'  Params: {params:,}')

    model, history, train_time = train_model(model, train_loader, val_loader, cfg, device)

    start = time.time()
    _, _, y_pred, y_true = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    test_time = time.time() - start
    metrics = compute_metrics(y_true, y_pred, meta['target_names'])

    gflops = ann_gflops(model, (cfg.n_components, cfg.window_size, cfg.window_size), device)
    # Profile over the whole test set, sharing the dispatch-level dense count so the
    # FLOPs column and the ANN energy reference come from a single source.
    energy = (profile_energy(model, test_loader, device, cfg, dense_gflops=gflops)
              if model_name in SNN_MODELS and not skip_energy else None)

    result = {
        'tag': tag, 'model': model_name, 'dataset': dataset, 'split': split, 'seed': seed,
        'attn_variant': cfg.attn_variant if model_name == 'SpikeSST' else None,
        'OA': metrics['OA'], 'AA': metrics['AA'], 'Kappa': metrics['Kappa'],
        'Macro_F1': metrics['Macro_F1'],
        'per_class': metrics['per_class'].tolist(),
        'params': params, 'ANN_GFLOPs': gflops,
        'train_time': train_time, 'test_time': test_time,
        'n_train': meta['n_train'], 'n_val': meta['n_val'], 'n_test': meta['n_test'],
        'timesteps': cfg.timesteps, 'window_size': cfg.window_size,
        'n_components': cfg.n_components, 'train_ratio': cfg.train_ratio,
        'use_sta': cfg.use_sta, 'use_sce': cfg.use_sce, 'augment': cfg.augment,
        'arch_variant': getattr(cfg, 'arch_variant', 'base'),
    }
    arch = _arch_config(cfg, model_name)
    result.update({
        'arch_config': arch,
        'arch_config_hash': _config_hash(arch),
    })
    if meta.get('split_manifest'):
        split_meta = meta['split_manifest']
        result.update({
            'split_hash': split_meta.get('sha256_16'),
            'split_manifest': split_meta,
        })
    if skip_energy and model_name in SNN_MODELS:
        # audit must treat this as "accuracy-only, energy pending", not as a row
        # that silently lacks energy fields
        result['energy_status'] = 'pending_profile'
    if energy:
        result.update({
            'energy_status': 'profiled',
            'E_SNN_uJ': energy['E_SNN_uJ'], 'E_ANN_uJ': energy['E_ANN_uJ'],
            'energy_ratio': energy['energy_ratio'], 'SOPs_M': energy['SOPs_M'],
            'MAC_ops_M': energy['MAC_ops_M'],
            'mean_firing_rate': energy['mean_firing_rate'],
            # audit trail: lets the reported energy be recomputed without retraining
            'n_profile_samples': energy['n_profile_samples'],
            'lif_rates': energy['lif_rates'],
            'per_layer': energy['per_layer'],
            'components': energy['components'],
        })
    print(f"  OA={metrics['OA']:.2f} AA={metrics['AA']:.2f} Kappa={metrics['Kappa']:.2f}"
          + (f"  E_ratio={energy['energy_ratio']:.4f} fr={energy['mean_firing_rate']:.3f}"
             if energy else ''), flush=True)

    ckpt_dir = os.path.join(out_dir, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{tag}.pt')
    torch.save(model.state_dict(), ckpt_path)
    result['checkpoint'] = os.path.relpath(ckpt_path, out_dir)
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)

    if plots:
        from spikesst.viz import (plot_confusion_matrix, plot_training_curves,
                                 plot_tsne, plot_segmentation_map)
        prefix = os.path.join(out_dir, tag)
        plot_training_curves(history, f'{prefix}_curves.png')
        plot_confusion_matrix(metrics['confusion_matrix'], meta['target_names'],
                              f'{prefix}_cm.png')
        if model_name in SNN_MODELS:
            feats, labels = extract_features(model, test_loader, device)
            plot_tsne(feats, labels, meta['target_names'], f'{prefix}_tsne.png')
            plot_segmentation_map(model, meta, cfg.window_size, device, f'{prefix}_seg.png')

    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['SpikeSST'])
    p.add_argument('--datasets', nargs='+', default=['IP'])
    p.add_argument('--splits', nargs='+', default=['random'])
    p.add_argument('--seeds', nargs='+', type=int, default=[345])
    p.add_argument('--attn-variant', default='sdsa', choices=['sdsa', 'additive'])
    p.add_argument('--timesteps', type=int, default=4)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--train-ratio', type=float, default=0.05)
    p.add_argument('--out', default=None)
    p.add_argument('--plots', action='store_true')
    p.add_argument('--no-sta', action='store_true')
    p.add_argument('--no-sce', action='store_true')
    p.add_argument('--augment', action='store_true')
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--window-size', type=int, default=9)
    p.add_argument('--n-components', type=int, default=15)
    p.add_argument('--plif', action='store_true')
    p.add_argument('--spectral-stem', action='store_true')
    p.add_argument('--arch-variant', default='base')
    p.add_argument('--split-manifest', default='')
    p.add_argument('--no-write-split-manifest', action='store_true')
    p.add_argument('--skip-energy', action='store_true',
                   help='skip ONLY energy profiling (all other audit keys are '
                        'still written); profile later with reprofile_energy.py')
    args = p.parse_args()

    cfg = Config(attn_variant=args.attn_variant, timesteps=args.timesteps,
                 epochs=args.epochs, train_ratio=args.train_ratio,
                 use_sta=not args.no_sta, use_sce=not args.no_sce,
                 augment=args.augment, patience=args.patience,
                 window_size=args.window_size, n_components=args.n_components,
                 plif=args.plif, spectral_stem=args.spectral_stem,
                 arch_variant=args.arch_variant,
                 split_manifest=args.split_manifest,
                 write_split_manifest=not args.no_write_split_manifest)
    out_dir = args.out or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    results, failures = [], []
    for dataset in args.datasets:
        for split in args.splits:
            for model_name in args.models:
                for seed in args.seeds:
                    try:
                        plots = args.plots and seed == args.seeds[0]
                        results.append(run_one(model_name, dataset, split, seed,
                                               cfg, device, out_dir, plots=plots,
                                               skip_energy=args.skip_energy))
                    except SplitManifestError as e:
                        # Never skip: a missing/mismatched split would silently
                        # drop the cell from the results table.
                        print(f'FATAL split manifest error for '
                              f'{dataset}/{model_name}/s{seed}: {e}', flush=True)
                        sys.exit(2)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        failures.append(f'{dataset}/{split}/{model_name}/s{seed}')

    # Aggregated summary (mean +/- std over seeds)
    if results:
        import pandas as pd
        df = pd.DataFrame(results)
        group_cols = ['dataset', 'split', 'model', 'attn_variant']
        agg = df.groupby(group_cols, dropna=False).agg(
            OA_mean=('OA', 'mean'), OA_std=('OA', 'std'),
            AA_mean=('AA', 'mean'), Kappa_mean=('Kappa', 'mean'),
            params=('params', 'first'), n=('OA', 'size'))
        print('\n' + agg.round(2).to_string())
        df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

    if failures:
        print(f'\n{len(failures)} cell(s) FAILED: ' + ', '.join(failures))
        sys.exit(1)


if __name__ == '__main__':
    main()
