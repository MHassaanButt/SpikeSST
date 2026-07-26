# SpikeSST: An Energy-Efficient Spiking Neural Network for Hyperspectral Image Classification

Official PyTorch implementation.

## Citation

```bibtex
@article{butt2026spikesst,
  title   = {SpikeSST: An Energy-Efficient Spiking Neural Network for
             Hyperspectral Image Classification},
  author  = {Butt, Muhammad Hassaan Farooq},
  year    = {2026}
}
```

## Requirements

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
```

## Data

```bash
export SNN_HSI_DATA=/path/to/hsi
```

The directory must contain:

```
PaviaU.mat             PaviaU_gt.mat
WHU_Hi_HanChuan.mat    WHU_Hi_HanChuan_gt.mat
QUH-Tangdaowan.mat     QUH-Tangdaowan_GT.mat
```

## Usage

Run step 1 first: it writes the split manifests that steps 2 and 3 are scored against.

```bash
SEEDS="345 445 545 645 745"
PROTOCOL="--splits disjoint --augment --patience 50 --window-size 13 --n-components 20"
```

```bash
# 1. official baselines (also writes results/splits/manifest.json)
git clone https://github.com/Katherine-Cao/HSI_SNN baselines/HSI_SNN
git clone https://github.com/Li-SNN/FRM-SNN        baselines/FRM-SNN

for ds in PU HC TD; do
  python train_baselines.py --baselines brain_inspired_snn frm_snn --datasets $ds \
      --epochs 100 --seeds $SEEDS $PROTOCOL --out results/baselines
done
```

```bash
# 2. SpikeSST, on the same splits
python train.py --models ASTST --datasets PU HC TD --seeds $SEEDS $PROTOCOL \
    --skip-energy --split-manifest results/splits/manifest.json \
    --no-write-split-manifest --out results/spikesst
```

```bash
# 3. energy profiles (forward-only, no retraining)
python profile_energy.py --src results/baselines --eval-batch-size 128
python profile_energy.py --src results/spikesst  --eval-batch-size 128
```

## Results

Overall accuracy (%), five seeds, spatially disjoint splits. Energy is estimated
arithmetic cost per patch at 45 nm, quoted on Tangdaowan.

| Method | Params | Energy | Pavia University | WHU-Hi-HanChuan | QUH-Tangdaowan |
|:--|--:|--:|--:|--:|--:|
| Brain-inspired SNN | 888 K | 599.4 µJ | 91.99 | 87.62 | 91.59 |
| FRM-SNN | 1 144 K | 3401.5 µJ | **94.33** | 88.65 | 91.99 |
| SpikeSST | **111 K** | **14.0 µJ** | 91.41 | **88.99** | **92.58** |

Per-class, AA and κ results are in the paper. All reported runs are in `results/`.

## Contact

Muhammad Hassaan Farooq Butt
