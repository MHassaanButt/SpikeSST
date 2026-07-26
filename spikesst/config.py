"""Experiment configuration."""
import os
from dataclasses import dataclass, field


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Override with the SNN_HSI_DATA env var if the datasets live elsewhere.
DEFAULT_DATA_PATH = os.environ.get(
    'SNN_HSI_DATA',
    os.path.normpath(os.path.join(REPO_ROOT, '..', '..', 'datasets', 'hsi_rs')))


@dataclass
class Config:
    # Dataset
    dataset: str = 'IP'                # 'IP', 'PU', 'HC', 'HH'
    data_path: str = DEFAULT_DATA_PATH

    # Preprocessing
    window_size: int = 9               # odd -> labeled pixel is the true patch center
    n_components: int = 15             # PCA components (fit on train pixels only)

    # Split
    split: str = 'random'              # 'random' | 'disjoint'
    train_ratio: float = 0.05
    val_ratio: float = 0.05
    seed: int = 345
    augment: bool = False              # random flip/rot90 on training patches
    disjoint_block: int = 16           # block size (px) for the spatially disjoint split
    split_manifest: str = ''           # manifest.json to load exact split coordinates from
    write_split_manifest: bool = True  # record generated split coordinates for auditing

    # Model
    model: str = 'SpikeSST'
    attn_variant: str = 'sdsa'         # 'sdsa' (spike-driven linear) | 'additive' (fixed additive)
    embed_dim: int = 64
    num_heads: int = 8
    num_layers: int = 2
    timesteps: int = 4
    dropout: float = 0.2
    use_sta: bool = True               # spiking temporal aggregation module
    use_sce: bool = True               # spiking channel enhancement module
    spectral_stem: bool = False        # SSFTT-style spectral 3D conv in the stem
    plif: bool = False                 # parametric LIF (learnable tau per layer)
    arch_variant: str = 'base'         # see models/__init__.py ARCH_VARIANTS

    # Training
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 30
    label_smoothing: float = 0.1
    warmup_epochs: int = 5
    amp: bool = True
    num_workers: int = 2
    eval_batch_size: int = 0   # 0 = auto (max(batch_size, 256)); set to cap profiling memory

    # Energy model (45 nm CMOS, standard in SNN literature)
    e_ac: float = 0.9e-12              # J per accumulate (spike-driven) op
    e_mac: float = 4.6e-12             # J per multiply-accumulate op

    # Output
    output_dir: str = os.path.join(REPO_ROOT, 'results', 'screen')


DATASET_INFO = {
    'IP': {
        'data_file': 'Indian_pines_corrected.mat', 'data_key': 'indian_pines_corrected',
        'gt_file': 'Indian_pines_gt.mat', 'gt_key': 'indian_pines_gt',
        'num_classes': 16,
        'target_names': ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
                         'Grass-pasture', 'Grass-trees', 'Grass-mowed',
                         'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                         'Soybean-clean', 'Wheat', 'Woods', 'Buildings', 'Stone-Steel'],
    },
    'PU': {
        'data_file': 'PaviaU.mat', 'data_key': 'paviaU',
        'gt_file': 'PaviaU_gt.mat', 'gt_key': 'paviaU_gt',
        'num_classes': 9,
        'target_names': ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted',
                         'Soil', 'Bitumen', 'Bricks', 'Shadows'],
    },
    'HC': {
        'data_file': 'WHU_Hi_HanChuan.mat', 'data_key': 'WHU_Hi_HanChuan',
        'gt_file': 'WHU_Hi_HanChuan_gt.mat', 'gt_key': 'WHU_Hi_HanChuan_gt',
        'num_classes': 16,
        'target_names': ['Strawberry', 'Cowpea', 'Soybean', 'Sorghum',
                         'Water spinach', 'Watermelon', 'Greens', 'Trees', 'Grass',
                         'Red roof', 'Gray roof', 'Plastic', 'Bare soil', 'Road',
                         'Bright object', 'Water'],
    },
    'PA': {   # QUH-Pingan (Haigang port, UAV-borne)
        'data_file': 'QUH-Pingan.mat', 'data_key': 'Haigang',
        'gt_file': 'QUH-Pingan_GT.mat', 'gt_key': 'HaigangGT',
        'num_classes': 10,
        # TODO: verify class-name order against the QUH benchmark paper
        'target_names': ['Ship', 'Seawater', 'Trees', 'Concrete floor',
                         'Floating pier', 'Brick houses', 'Steel houses',
                         'Wharf construction land', 'Car', 'Road'],
    },
    'TD': {   # QUH-Tangdaowan (coastal park, UAV-borne)
        'data_file': 'QUH-Tangdaowan.mat', 'data_key': 'Tangdaowan',
        'gt_file': 'QUH-Tangdaowan_GT.mat', 'gt_key': 'TangdaowanGT',
        'num_classes': 18,
        # TODO: verify class-name order against the QUH benchmark paper
        'target_names': ['Rubber track', 'Flaggingstone', 'Sandy', 'Asphalt',
                         'Boardwalk', 'Rocky shallows', 'Grassland', 'Bulrush',
                         'Gravel road', 'Ligustrum vicaryi', 'Coniferous pine',
                         'Spiraea', 'Bare soil', 'Buxus sinica',
                         'Photinia serrulata', 'Populus', 'Ulmus pumila',
                         'Seawater'],
    },
    'HH': {
        'data_file': 'WHU_Hi_HongHu.mat', 'data_key': 'WHU_Hi_HongHu',
        'gt_file': 'WHU_Hi_HongHu_gt.mat', 'gt_key': 'WHU_Hi_HongHu_gt',
        'num_classes': 22,
        'target_names': ['Red roof', 'Road', 'Bare soil', 'Cotton',
                         'Cotton firewood', 'Rape', 'Chinese cabbage',
                         'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis',
                         'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa',
                         'Celtuce', 'Film covered lettuce', 'Romaine lettuce',
                         'Carrot', 'White radish', 'Garlic sprout', 'Broad bean', 'Tree'],
    },
}
