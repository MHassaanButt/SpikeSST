"""Shared spiking building blocks.

Conventions (all multi-step, true temporal dynamics):
- Activations flow as (T, B, N, D) for token streams and (T, B, C, H, W) for conv maps.
- Every LIFNode runs in SpikingJelly multi-step mode ('m'), so membrane potential
  integrates across the T timesteps (this was the core bug in the v1 code, which
  folded T into the batch and called single-step neurons once).
- "Pre-spike" design: each module first converts its float residual input to binary
  spikes via BN+LIF, so every weight layer (except the analog stem conv1 and any ANN
  baseline) consumes binary tensors -> accumulate-only ops on neuromorphic hardware.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate

_NEURON_KIND = 'lif'


def set_neuron(kind):
    """Select the spiking neuron for subsequently built modules: 'lif' | 'plif'.
    PLIF learns tau per layer (one extra parameter each)."""
    global _NEURON_KIND
    assert kind in ('lif', 'plif')
    _NEURON_KIND = kind


def lif(v_threshold=1.0, tau=2.0):
    common = dict(v_threshold=v_threshold, decay_input=False, detach_reset=True,
                  surrogate_function=surrogate.ATan(), step_mode='m')
    if _NEURON_KIND == 'plif':
        return neuron.ParametricLIFNode(init_tau=tau, **common)
    return neuron.LIFNode(tau=tau, **common)


class TokenBN(nn.Module):
    """BatchNorm1d over the channel dim of a (T, B, N, D) token stream."""

    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, N, D = x.shape
        return self.bn(x.reshape(-1, D)).reshape(T, B, N, D)


class SpikeInput(nn.Module):
    """BN + LIF converting a float residual stream to binary spikes."""

    def __init__(self, dim):
        super().__init__()
        self.bn = TokenBN(dim)
        self.lif = lif()

    def forward(self, x):
        return self.lif(self.bn(x))


class SpikingConvStem(nn.Module):
    """Direct input encoding: analog patch -> conv1 -> BN -> LIF (first spike layer).

    conv1 consumes the analog input (MAC ops, computed once since the input is
    identical across timesteps); conv2 consumes spikes (AC ops).

    With `spectral=True`, a grouped spectral 3-D convolution (SSFTT-style) is
    applied to the analog cube before the spatial conv, sharpening spectral
    features at a small parameter cost. It runs once per sample (analog input),
    so it is charged at E_MAC like conv1.
    """

    def __init__(self, in_channels, embed_dim, spectral=False, first_kernel=3,
                 n_layers=2, depthwise_spatial=False):
        super().__init__()
        self.spectral = spectral
        self.depthwise_spatial = depthwise_spatial
        c_in = in_channels
        if spectral:
            self.spec_conv = nn.Conv3d(1, 4, kernel_size=(7, 1, 1), padding=(3, 0, 0))
            self.spec_bn = nn.BatchNorm3d(4)
            c_in = 4 * in_channels
        # first_kernel=1 keeps the (analog, E_MAC-charged) encoding convolution
        # cheap: it then mixes spectra only, and spatial reach comes from the
        # spike-driven layers that follow.
        self.conv1 = nn.Conv2d(c_in, embed_dim // 2, first_kernel,
                               padding=first_kernel // 2)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        self.lif1 = lif()
        self.dw_spatial = None
        if depthwise_spatial:
            self.dw_spatial = nn.Conv2d(embed_dim // 2, embed_dim // 2, 3,
                                        padding=1, groups=embed_dim // 2,
                                        bias=False)
            self.dw_bn = nn.BatchNorm2d(embed_dim // 2)
            self.dw_lif = lif()
        self.conv2 = nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.lif2 = lif()
        self.extra = None
        if n_layers > 2:                       # optional third spiking conv
            self.extra = nn.ModuleList()
            for _ in range(n_layers - 2):
                self.extra.append(nn.ModuleDict({
                    'conv': nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                    'bn': nn.BatchNorm2d(embed_dim),
                }))
            self.extra_lifs = nn.ModuleList([lif() for _ in range(n_layers - 2)])

    def forward(self, x):
        # x: (T, B, C, H, W)
        T, B = x.shape[:2]
        h = x.flatten(0, 1)
        if self.spectral:
            h = torch.relu(self.spec_bn(self.spec_conv(h.unsqueeze(1))))
            h = h.flatten(1, 2)  # (TB, 4*C, H, W), still analog
        h = self.bn1(self.conv1(h))
        h = self.lif1(h.reshape(T, B, *h.shape[1:]))
        if self.dw_spatial is not None:
            y = self.dw_bn(self.dw_spatial(h.flatten(0, 1)))
            h = self.dw_lif(y.reshape(T, B, *y.shape[1:]))
        h = self.bn2(self.conv2(h.flatten(0, 1)))
        h = self.lif2(h.reshape(T, B, *h.shape[1:]))
        if self.extra is not None:
            for blk, lf in zip(self.extra, self.extra_lifs):
                y = blk['bn'](blk['conv'](h.flatten(0, 1)))
                h = lf(y.reshape(T, B, *y.shape[1:]))
        return h


class SpikingFFN(nn.Module):
    """Pre-spike feed-forward. Dropout on the float output only (never on spikes)."""

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.spike_in = SpikeInput(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.bn1 = TokenBN(hidden_dim)
        self.lif1 = lif()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.bn2 = TokenBN(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        s = self.spike_in(x)
        h = self.lif1(self.bn1(self.fc1(s)))
        return self.dropout(self.bn2(self.fc2(h)))


class DepthwiseSpikeMix(nn.Module):
    """Local spatial mixing over the token grid, in the spiking domain.

    Spike-driven linear attention reduces the keys and values to one context
    vector that is broadcast identically to every token, so it contributes no
    token-to-token spatial differentiation: beyond the receptive field of the
    convolutional stem, spatial information only enters through permutation-
    invariant pooling. This module restores local structure with a depthwise
    convolution over the (w x w) token grid. Its input is binary (pre-spike
    design), so the convolution is accumulate-only, and being depthwise it costs
    D*k^2 parameters -- about 1% of the model.
    """

    def __init__(self, embed_dim, window_size, kernel_size=3, init_scale=1.0):
        super().__init__()
        self.w = window_size
        self.spike_in = SpikeInput(embed_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size,
                              padding=kernel_size // 2, groups=embed_dim, bias=False)
        self.bn = TokenBN(embed_dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        T, B, N, D = x.shape
        s = self.spike_in(x)                                   # binary tokens
        s = s.reshape(T * B, self.w, self.w, D).permute(0, 3, 1, 2)
        s = self.conv(s)
        s = s.permute(0, 2, 3, 1).reshape(T, B, N, D)
        return self.scale * self.bn(s)


class SpikingTemporalAggregation(nn.Module):
    """Learned mixing across the T timesteps (T*T params). Meaningful only because
    timesteps now carry genuinely different spike patterns."""

    def __init__(self, embed_dim, timesteps):
        super().__init__()
        self.spike_in = SpikeInput(embed_dim)
        self.temporal_mix = nn.Linear(timesteps, timesteps, bias=False)
        self.bn = TokenBN(embed_dim)

    def forward(self, x):
        s = self.spike_in(x)                       # (T, B, N, D) spikes
        y = self.temporal_mix(s.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
        return self.bn(y)


class SpikingChannelEnhancement(nn.Module):
    """Squeeze-excite with a binary spiking gate (mask instead of sigmoid multiply)."""

    def __init__(self, embed_dim, reduction=4):
        super().__init__()
        hidden = max(embed_dim // reduction, 16)
        self.spike_in = SpikeInput(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.bn_h = nn.BatchNorm1d(hidden)
        self.lif_h = lif()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.bn_g = nn.BatchNorm1d(embed_dim)
        self.lif_g = lif()
        self.bn_out = TokenBN(embed_dim)

    def forward(self, x):
        T, B, N, D = x.shape
        s = self.spike_in(x)
        p = s.mean(dim=2)                          # (T, B, D) firing rates
        h = self.bn_h(self.fc1(p).reshape(-1, self.fc1.out_features)).reshape(T, B, -1)
        h = self.lif_h(h)
        g = self.bn_g(self.fc2(h).reshape(-1, D)).reshape(T, B, D)
        g = self.lif_g(g)                          # binary channel gate
        return self.bn_out(s * g.unsqueeze(2))     # spike AND gate
