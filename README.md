# Energy-Efficient Spiking Transformer for HSI Classification

Rebuilt pipeline (v3). `main.py` is the old notebook-style code, kept only as
reference — it contains fatal bugs (no temporal dynamics, input-destroying encoder,
rank-1 attention, broken energy accounting) and should not be used.

## Layout

```
snn_hsi/
  config.py        dataclass config + dataset registry (data at ../datasets)
  data.py          PCA (train-pixel fit), random + spatially disjoint splits, lazy patches
  models/
    layers.py      multi-step LIF blocks (pre-spike design: weights consume binary spikes)
    attention.py   variant A 'additive' (fixed co-activation) | variant B 'sdsa' (linear, O(N))
    astst.py       proposed model
    baselines.py   3D-CNN, SpectralFormer, SSFTT, SSMamba, Spikformer (multi-step)
  engine.py        train/eval, metrics (OA/AA/Kappa/F1, zero-support-safe)
  energy.py        SOP-based energy: E = E_MAC*analog + E_AC*sum(fr*FLOPs), 45nm constants
  viz.py           confusion matrix, t-SNE, curves, full-scene maps
scripts/
  sanity_checks.py CPU checks that the v1 bugs stay fixed (run after any model change)
  run_experiment.py  resumable CLI (per-combo JSON in results dir)
  slurm_run.sh     sbatch wrapper (GPU partition)
  run_all.sh       staged experiment suite
```

## Environment

`~/venvs/snn` (Python 3.9, torch 2.8+cu128 for the RTX Pro 6000 node, SpikingJelly).

## Usage

```bash
# sanity checks (login node, CPU)
~/venvs/snn/bin/python scripts/sanity_checks.py

# single experiment on the GPU node
sbatch scripts/slurm_run.sh --models ASTST --attn-variant sdsa \
    --datasets IP PU --splits random disjoint --seeds 345 445 545 --plots

# everything
bash scripts/run_all.sh
```

Finished (model, dataset, split, seed) combos are skipped on re-run via their JSON
files, so jobs are safe to resubmit.
