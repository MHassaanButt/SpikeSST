"""Spiking attention variants.

Both variants operate on (T, B, N, D) float residual streams, spike them internally
(pre-spike design), and return a float residual contribution. Both are
multiplication-free at inference in the sense that every product involves at least
one binary operand (mask + accumulate on neuromorphic hardware).

Variant A ("additive", fixed): the original Q+K idea repaired. For binary spikes a
thresholded sum h(q_id + k_jd) with threshold in (1, 2) is exactly co-activation
AND(q_id, k_jd), so the per-dim weighted score
    attn[i, j] = sum_d w_d * AND(q_id, k_jd)
is computed as (q * w) @ k^T -- genuinely pairwise (fixes the rank-1 bug in v1)
while remaining a pure spike-count accumulation. O(N^2 d).

Variant B ("sdsa", spike-driven linear attention): SDSA-style,
    out = q  *  SN(sum_j k_j * v_j)
Hadamard products of binary tensors + a column accumulation -> O(N d), no attention
matrix at all. Strongest energy profile.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import lif, SpikeInput, TokenBN


class _QKVProj(nn.Module):
    """Shared spike-in + spiking Q/K/V projections with multi-head reshape."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.spike_in = SpikeInput(embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_bn, self.k_bn, self.v_bn = TokenBN(embed_dim), TokenBN(embed_dim), TokenBN(embed_dim)
        self.q_lif, self.k_lif, self.v_lif = lif(), lif(), lif()

    def forward(self, x):
        T, B, N, D = x.shape
        s = self.spike_in(x)
        q = self.q_lif(self.q_bn(self.q_proj(s)))
        k = self.k_lif(self.k_bn(self.k_proj(s)))
        v = self.v_lif(self.v_bn(self.v_proj(s)))
        # (T, B, N, D) -> (T, B, H, N, d)
        def heads(t):
            return t.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        return heads(q), heads(k), heads(v)


class AdditiveSpikingAttention(nn.Module):
    """Variant A: per-dim weighted co-activation scores (fixed additive attention)."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.qkv = _QKVProj(embed_dim, num_heads)
        d = self.qkv.head_dim
        # positive per-head per-dim weights; init so scores start near mean pooling
        self.dim_weight = nn.Parameter(torch.zeros(num_heads, 1, d))
        self.attn_lif = lif(v_threshold=0.5)
        self.out_bn = TokenBN(embed_dim)
        self.out_lif = lif()
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_bn = TokenBN(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T, B, N, D = x.shape
        q, k, v = self.qkv(x)                                # (T, B, H, N, d) spikes
        w = F.softplus(self.dim_weight)                      # (H, 1, d) > 0
        attn = torch.matmul(q * w, k.transpose(-2, -1))      # sum_d w_d AND(q, k)
        attn = self.attn_lif(attn)                           # binary attention map
        y = torch.matmul(attn, v)                            # spike-count aggregation
        y = y.permute(0, 1, 3, 2, 4).reshape(T, B, N, D)
        y = self.out_lif(self.out_bn(y))                     # back to spikes
        return self.dropout(self.proj_bn(self.out_proj(y)))


class SpikeDrivenAttention(nn.Module):
    """Variant B: SDSA-style linear spike attention, O(N d)."""

    def __init__(self, embed_dim, num_heads, dropout=0.1, local_gate=0,
                 window_size=9):
        super().__init__()
        self.qkv = _QKVProj(embed_dim, num_heads)
        self.kv_bn = TokenBN(embed_dim)
        self.kv_lif = lif()
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_bn = TokenBN(embed_dim)
        self.dropout = nn.Dropout(dropout)
        # local_gate > 0 pools the key-value context over a (local_gate x
        # local_gate) neighbourhood of the token grid instead of globally, so the
        # gate varies per token. Still O(N d): the pooling is a fixed-size sum.
        self.local_gate = local_gate
        self.w = window_size
        self._profile = False
        self._kv_rate_sum = 0.0
        self._kv_calls = 0

    def start_profile(self):
        """Record the measured joint (k AND v) co-activity for the energy model,
        rather than approximating it by the product of marginal firing rates."""
        self._profile = True
        self._kv_rate_sum = 0.0
        self._kv_calls = 0

    def stop_profile(self):
        self._profile = False

    @property
    def kv_rate(self):
        """Measured mean of (k AND v); None if never profiled."""
        if self._kv_calls == 0:
            return None
        return self._kv_rate_sum / self._kv_calls

    def forward(self, x):
        T, B, N, D = x.shape
        q, k, v = self.qkv(x)                                # (T, B, H, N, d) spikes
        kv_and = k * v                                       # logical AND of two spike maps
        if self._profile:
            self._kv_rate_sum += float(kv_and.detach().float().mean())
            self._kv_calls += 1
        if self.local_gate:
            # per-token context: sum (k AND v) over a local neighbourhood of the
            # token grid, so the gate carries spatial structure
            k_sz, w = self.local_gate, self.w
            sp = kv_and.permute(0, 1, 2, 4, 3).reshape(T * B * self.qkv.num_heads,
                                                       self.qkv.head_dim, w, w)
            sp = F.avg_pool2d(sp, k_sz, stride=1, padding=k_sz // 2) * (k_sz ** 2)
            kv = sp.reshape(T, B, self.qkv.num_heads, self.qkv.head_dim, N) \
                   .permute(0, 1, 4, 2, 3).reshape(T, B, N, D)
        else:
            kv = kv_and.sum(dim=3, keepdim=True)             # accumulate: (T,B,H,1,d)
            kv = kv.permute(0, 1, 3, 2, 4).reshape(T, B, 1, D)
        kv = self.kv_lif(self.kv_bn(kv))                     # binary context
        qf = q.permute(0, 1, 3, 2, 4).reshape(T, B, N, D)
        y = qf * kv                                          # binary Hadamard (mask)
        return self.dropout(self.proj_bn(self.out_proj(y)))


def build_attention(variant, embed_dim, num_heads, dropout, local_gate=0,
                    window_size=9):
    if variant == 'additive':
        return AdditiveSpikingAttention(embed_dim, num_heads, dropout)
    if variant == 'sdsa':
        return SpikeDrivenAttention(embed_dim, num_heads, dropout,
                                    local_gate=local_gate, window_size=window_size)
    raise ValueError(f'Unknown attention variant: {variant}')
