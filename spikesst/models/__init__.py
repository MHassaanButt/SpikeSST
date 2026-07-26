from .spikesst import SpikeSST

# Architecture variants screened on the held-out development scene. Each entry
# overrides SpikeSST constructor arguments; 'base' is the published configuration.
ARCH_VARIANTS = {
    'base':          {},
    'center_readout': {'readout': 'center_pool'},
    'center_dsm3':   {'readout': 'center_pool', 'dsm_kernel': 3,
                      'dsm_position': 'both', 'dsm_init_scale': 1e-3},
    'center_dsm3_slimstem': {
        'readout': 'center_pool', 'dsm_kernel': 3, 'dsm_position': 'both',
        'dsm_init_scale': 1e-3, 'stem_first_kernel': 1,
        'stem_depthwise_spatial': True,
    },
    'dsm3':          {'dsm_kernel': 3},
    'dsm5':          {'dsm_kernel': 5},
    'dsm3_slimstem': {'dsm_kernel': 3, 'stem_first_kernel': 1},
    'deepstem':      {'stem_layers': 3},
    'wide96':        {'embed_dim': 96},
    'deep3':         {'num_layers': 3},
    'localattn':     {'local_gate': 3},
}


def build_model(name, in_channels, num_classes, cfg):
    if name == 'SpikeSST':
        kwargs = dict(embed_dim=cfg.embed_dim,
                      num_heads=cfg.num_heads, num_layers=cfg.num_layers,
                      timesteps=cfg.timesteps, window_size=cfg.window_size,
                      dropout=cfg.dropout, attn_variant=cfg.attn_variant,
                      use_sta=cfg.use_sta, use_sce=cfg.use_sce,
                      spectral_stem=getattr(cfg, 'spectral_stem', False),
                      plif=getattr(cfg, 'plif', False))
        variant = getattr(cfg, 'arch_variant', 'base')
        if variant not in ARCH_VARIANTS:
            raise ValueError(f'Unknown arch variant: {variant}')
        kwargs.update(ARCH_VARIANTS[variant])
        return SpikeSST(in_channels, num_classes, **kwargs)
    raise ValueError(f'Unknown model: {name}')
