# SpikeSST: An Energy-Efficient Spiking Neural Network for Hyperspectral Image Classification



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

```bash
SEEDS="345 445 545 645 745"
PROTOCOL="--splits disjoint --augment --patience 50 --window-size 13 --n-components 20"
```


```bash
# SpikeSST, on the same splits
python train.py --models SpikeSST --datasets PU HC TD --seeds $SEEDS $PROTOCOL \
    --skip-energy --split-manifest results/splits/manifest.json \
    --no-write-split-manifest --out results/spikesst
```

```bash
# energy profiles (forward-only, no retraining)
python profile_energy.py --src results/baselines --eval-batch-size 128
python profile_energy.py --src results/spikesst  --eval-batch-size 128
```

## Results

Overall accuracy (%), five seeds, spatially disjoint splits. Energy is estimated
arithmetic cost per patch at 45 nm, quoted on Tangdaowan.

| Method | Params | Energy | Pavia University | WHU-Hi-HanChuan | QUH-Tangdaowan |
|:--|--:|--:|--:|--:|--:|
| Brain-inspired SNN | 0.888 M | 599.4 µJ | 91.99 | 87.62 | 91.59 |
| FRM-SNN | 1.144 M | 3401.5 µJ | **94.33** | 88.65 | 91.99 |
| SpikeSST | **111 K** | **14.0 µJ** | 91.41 | **88.99** | **92.58** |

Per-class, AA and κ results are in the paper. All reported runs are in `results/`.

