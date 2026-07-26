"""SpikeSST: a spike-driven spectral-spatial network for HSI classification.

Fixes over v2 (main.py):
- true multi-step temporal dynamics (LIF membranes integrate across T)
- direct analog input encoding (no information-destroying encoder LIF)
- non-degenerate attention (two variants, see attention.py)
- spike-compatible SE gating, no dropout on spikes, spike-consuming classifier head
"""
import torch
import torch.nn as nn

from .attention import build_attention
from .layers import (DepthwiseSpikeMix, SpikeInput, SpikingConvStem, SpikingFFN,
                     SpikingTemporalAggregation, SpikingChannelEnhancement,
                     set_neuron)


class SpikeSSTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, timesteps, dropout=0.2,
                 attn_variant='sdsa', use_sta=True, use_sce=True,
                 dsm_kernel=0, dsm_init_scale=1.0, dsm_position='post',
                 local_gate=0, window_size=9):
        super().__init__()
        self.dsm_pre = (DepthwiseSpikeMix(embed_dim, window_size, dsm_kernel,
                                          init_scale=dsm_init_scale)
                        if dsm_kernel and dsm_position in ('pre', 'both') else None)
        self.attn = build_attention(attn_variant, embed_dim, num_heads, dropout,
                                    local_gate=local_gate, window_size=window_size)
        self.dsm = (DepthwiseSpikeMix(embed_dim, window_size, dsm_kernel,
                                      init_scale=dsm_init_scale)
                    if dsm_kernel and dsm_position in ('post', 'both') else None)
        self.sta = SpikingTemporalAggregation(embed_dim, timesteps) if use_sta else None
        self.sce = SpikingChannelEnhancement(embed_dim) if use_sce else None
        self.ffn = SpikingFFN(embed_dim, embed_dim * 2, dropout)

    def forward(self, x):
        if self.dsm_pre is not None:
            x = x + self.dsm_pre(x)
        x = x + self.attn(x)
        if self.dsm is not None:
            x = x + self.dsm(x)
        if self.sta is not None:
            x = x + self.sta(x)
        if self.sce is not None:
            x = x + self.sce(x)
        return x + self.ffn(x)


class SpikeSST(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=64, num_heads=8,
                 num_layers=2, timesteps=4, window_size=9, dropout=0.2,
                 attn_variant='sdsa', use_sta=True, use_sce=True,
                 spectral_stem=False, plif=False,
                 dsm_kernel=0, dsm_init_scale=1.0, dsm_position='post',
                 local_gate=0, stem_first_kernel=3, stem_layers=2,
                 stem_depthwise_spatial=False, readout='mean'):
        super().__init__()
        self.timesteps = timesteps
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.readout = readout
        set_neuron('plif' if plif else 'lif')

        self.conv_stem = SpikingConvStem(in_channels, embed_dim,
                                         spectral=spectral_stem,
                                         first_kernel=stem_first_kernel,
                                         n_layers=stem_layers,
                                         depthwise_spatial=stem_depthwise_spatial)
        self.num_tokens = window_size * window_size
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.num_tokens, embed_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SpikeSSTBlock(embed_dim, num_heads, timesteps, dropout,
                       attn_variant, use_sta, use_sce,
                       dsm_kernel=dsm_kernel, dsm_init_scale=dsm_init_scale,
                       dsm_position=dsm_position, local_gate=local_gate,
                       window_size=window_size)
            for _ in range(num_layers)
        ])

        self.spike_out = SpikeInput(embed_dim)
        if readout == 'center_pool':
            self.head = nn.Linear(embed_dim * 2, num_classes)
        elif readout == 'mean':
            self.head = nn.Linear(embed_dim, num_classes)
        else:
            raise ValueError(f'Unknown readout: {readout}')
        set_neuron('lif')  # don't leak the neuron choice into later-built models

    def forward(self, x, return_features=False):
        B = x.shape[0]
        T = self.timesteps
        x = x.unsqueeze(0).repeat(T, 1, 1, 1, 1)          # (T, B, C, H, W), analog
        x = self.conv_stem(x)                             # (T, B, D, H, W), spikes
        x = x.flatten(3).transpose(2, 3)                  # (T, B, N, D)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        s = self.spike_out(x)                             # (T, B, N, D) spikes
        if self.readout == 'center_pool':
            center = (self.window_size * self.window_size) // 2
            features = torch.cat([s[:, :, center, :].mean(dim=0),
                                  s.mean(dim=(0, 2))], dim=-1)
            logits = self.head(features)
        else:
            logits = self.head(s).mean(dim=(0, 2))        # rate decoding over T and tokens
        if return_features:
            return logits, s.mean(dim=(0, 2))
        return logits
