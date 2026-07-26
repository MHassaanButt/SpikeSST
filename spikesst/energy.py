"""Arithmetic energy and complexity accounting (Spike-driven-Transformer
methodology).

Per weight layer l:  SOPs_l = fr_l x FLOPs_l  where fr_l is the measured firing rate
of the (binary) tensor feeding the layer and FLOPs_l includes all spatial/token/time
dims (this fixes the v1 code, which counted a single kernel application per layer and
raw spike counts instead of synaptic operations).

Energy model (45 nm):
  E_SNN = E_MAC x FLOPs_analog  +  E_AC x sum_l SOPs_l
    - the analog stem conv is computed once (its input is identical across T)
    - float-input layers that remain (e.g. the tiny SE squeeze fc) are counted at E_MAC
    - functional attention matmuls (not nn.Module hooks) are added analytically from
      the measured Q/K/V/attn firing rates
  E_ANN = E_MAC x FLOPs of the same architecture run as a one-step ANN.
BN (foldable into weights) and neuron-update costs are excluded, as is standard.
"""
import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

# Bump when the accounting rules change; stored in every profile so stale numbers
# can be detected by the audit rather than silently mixed with current ones.
ENERGY_ACCOUNTING_VERSION = 2


def _is_binary(x, eps=1e-6):
    """True when every element is (numerically) 0 or 1, i.e. a spike tensor.

    A small tolerance guards against float noise; it is not expected to reclassify
    much, because the external baselines emit exact 0/1 from `input.gt(0).float()`
    and profiling runs without autocast. Layers that fail this test are genuinely
    real-valued (batch-norm outputs, residual/membrane sums, pooled activations).
    """
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        return False
    return bool(((x.abs() < eps) | ((x - 1).abs() < eps)).all())


def _kernel_ops(module):
    if isinstance(module, nn.Linear):
        return module.in_features
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        ops = module.in_channels // module.groups
        for k in module.kernel_size:
            ops *= k
        return ops
    return None


def _component_of(name):
    """Map a layer name to the architectural component it belongs to, for the
    per-component energy breakdown reported in the paper.

    Covers our own module names first, then generic patterns so that external
    baselines (whose layers are named conv1/conv3_1/fc1/...) do not all collapse
    into a single 'other' bucket.
    """
    n = name.lower()
    # --- our architecture ---
    if 'conv_stem' in n:
        return 'stem'
    if '.sce.' in n or n.endswith('sce'):
        return 'sce'
    if '.sta.' in n or n.endswith('sta'):
        return 'sta'
    if '.dsm' in n or n.endswith('dsm'):
        return 'dsm'
    if 'attn' in n or 'qkv' in n:
        return 'attention'
    if 'ffn' in n:
        return 'ffn'
    if 'head' in n:
        return 'classifier'
    # --- generic / external baselines ---
    if n.startswith('fc') or 'classifier' in n or 'linear' in n or n.endswith('.fc'):
        return 'classifier'
    if 'shortcut' in n or 'downsample' in n or 'residual' in n:
        return 'shortcut'
    if 'pool' in n:
        return 'pool'
    if n in ('conv1', 'conv_1') or n.startswith('conv1.'):
        return 'stem'          # first (analog-input) convolution
    if 'conv' in n:
        return 'conv'
    return 'other'


@torch.no_grad()
def profile_energy(model, loader, device, cfg, max_batches=None, dense_gflops=None):
    """Measure firing rates over the loader (the whole test set by default) and
    convert them to an estimated arithmetic energy.

    This is an operation-count model: it accounts for the multiply-accumulate and
    accumulate work of weighted layers plus the functional attention ops, and
    excludes memory access, neuron-state updates, control logic, interconnect,
    PCA/patch extraction, and off-chip data movement. The analog stem convolution
    sees an identical input at every timestep, so it is charged once (a caching
    assumption about the deployment target; the PyTorch code recomputes it).
    """
    model.eval()
    T = getattr(model, 'timesteps', 1)

    layer_stats = {}   # name -> dict(macs, binary, rate_sum, calls, t_replicated)
    spike_rates = {}   # name -> dict(rate_sum, calls)  (duck-typed, any neuron class)
    hooks = []
    cur_batch = {'n': 0}   # samples in the batch currently being profiled

    def make_layer_hook(name, module):
        kops = _kernel_ops(module)

        def hook(mod, inputs, output):
            x = inputs[0]
            st = layer_stats.setdefault(name, {'macs': 0.0, 'binary': True,
                                               'rate_sum': 0.0, 'calls': 0,
                                               't_replicated': False})
            st['macs'] += output.numel() * kops
            st['binary'] = st['binary'] and _is_binary(x)
            st['rate_sum'] += float(x.float().mean())
            st['calls'] += 1
            # Whether this layer sees a T-replicated tensor (leading dim T*B) or a
            # plain per-sample batch. This is what decides the static-input rule:
            # our stem is invoked once with a (T*B, ...) tensor so its op count is
            # inflated T-fold, whereas an external model that computes
            # `static_input = conv1(x)` outside its timestep loop is already
            # counted once and must NOT be divided again.
            if T > 1 and cur_batch['n'] and x.dim() >= 1:
                st['t_replicated'] = (x.shape[0] == T * cur_batch['n'])
        return hook

    def make_spike_hook(name):
        def hook(mod, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            if not _is_binary(output):
                return          # not a spiking output; ignore
            st = spike_rates.setdefault(name, {'rate_sum': 0.0, 'calls': 0})
            st['rate_sum'] += float(output.float().mean())
            st['calls'] += 1
        return hook

    # Duck-typed spiking-neuron detection: external baselines define their own
    # neuron classes, so an isinstance(LIFNode) check finds nothing and leaves the
    # firing rate reading a meaningless 0. Instead hook every leaf module and keep
    # the ones that actually emit binary tensors.
    for name, m in model.named_modules():
        if _kernel_ops(m) is not None:
            hooks.append(m.register_forward_hook(make_layer_hook(name, m)))
        elif isinstance(m, neuron.BaseNode) or (
                len(list(m.children())) == 0 and not isinstance(
                    m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                        nn.Dropout, nn.Identity))):
            hooks.append(m.register_forward_hook(make_spike_hook(name)))

    # enable joint-coactivity recording inside spike-driven attention
    profiled_attn = [m for m in model.modules() if hasattr(m, 'start_profile')]
    for m in profiled_attn:
        m.start_profile()

    n_samples = 0
    for i, (x, _) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        cur_batch['n'] = x.shape[0]
        model(x.to(device))
        functional.reset_net(model)
        n_samples += x.shape[0]

    for h in hooks:
        h.remove()
    for m in profiled_attn:
        m.stop_profile()

    rates = {k: v['rate_sum'] / max(v['calls'], 1) for k, v in spike_rates.items()}

    e_ac, e_mac = cfg.e_ac, cfg.e_mac
    e_snn, e_ann, total_sops, total_ann_flops = 0.0, 0.0, 0.0, 0.0
    total_mac_ops = 0.0                       # real-valued MAC work retained in the SNN
    per_layer = {}
    components = {}                           # component -> dict(sops, macs, energy_J)

    def bump(name, sops=0.0, macs=0.0, energy=0.0):
        c = components.setdefault(_component_of(name),
                                  {'SOPs': 0.0, 'MACs': 0.0, 'energy_J': 0.0})
        c['SOPs'] += sops
        c['MACs'] += macs
        c['energy_J'] += energy

    # Three explicit buckets, never silently merged. `e_opt` accumulates an
    # OPTIMISTIC LOWER BOUND in which multibit layers are charged at the cheaper
    # AC rate; it is reported beside the primary number, never in place of it.
    binary_ac_J = multibit_mac_J = static_input_mac_J = classifier_J = 0.0
    e_opt = 0.0
    for name, st in layer_stats.items():
        macs = st['macs'] / n_samples            # per sample, T included via tensor dims
        rate = st['rate_sum'] / max(st['calls'], 1)
        comp = _component_of(name)
        if st['binary']:
            kind = 'binary'
            ops = rate * macs
            e = ops * e_ac
            total_sops += ops
            bump(name, sops=ops, energy=e)
            binary_ac_J += e
            e_opt += e
        elif st['t_replicated']:
            # Analog input replicated across timesteps by our implementation: the
            # op count is inflated T-fold, hardware computes it once. External
            # models that hoist this out of their timestep loop are already
            # counted once (t_replicated False) and must not be divided again.
            kind = 'static_analog'
            ops = macs / T
            e = ops * e_mac
            total_mac_ops += ops
            bump(name, macs=ops, energy=e)
            static_input_mac_J += e
            e_opt += ops * e_ac
        else:
            # Genuinely real-valued input (batch-norm output, residual/membrane
            # sum, pooled activation, or an analog input conv already computed
            # once). Charged at the full MAC rate -- this is the primary rule and
            # multibit activity is never reclassified as AC.
            kind = 'multibit'
            e = macs * e_mac
            total_mac_ops += macs
            bump(name, macs=macs, energy=e)
            multibit_mac_J += e
            e_opt += macs * e_ac
        if comp == 'classifier':
            classifier_J += e
        e_snn += e
        total_ann_flops += macs / T
        per_layer[name] = {'macs_per_sample': macs, 'binary': st['binary'],
                           'input_kind': kind, 'rate': rate, 'component': comp,
                           't_replicated': st['t_replicated'], 'energy_J': e}

    # Functional attention ops (Hadamard AND / matmul) added analytically, using the
    # measured joint co-activity where two spike tensors are multiplied together.
    attn_sops, attn_ann_flops = _attention_ops(model, rates, T)
    e_attn = attn_sops * e_ac
    e_snn += e_attn
    e_opt += e_attn
    binary_ac_J += e_attn
    total_sops += attn_sops
    total_ann_flops += attn_ann_flops
    bump('attention.functional', sops=attn_sops, energy=e_attn)

    # Mean firing rate. Module-level hooks only see spikes when the neuron is a
    # Module; the external baselines emit theirs from a functional autograd call
    # (`SpikeAct.apply`), so nothing is captured for them. The rate is measured
    # anyway as the mean of each binary tensor feeding a weighted layer, so fall
    # back to that -- architecture-agnostic and never silently zero.
    # Duck-typing also matches modules whose output is all zeros (trivially
    # binary), e.g. a pooling layer downstream of a silent branch. Such a
    # detection is degenerate, not a measurement of 0 Hz firing, so require at
    # least one strictly positive rate before trusting the module-level view.
    binary_rates = [v['rate'] for v in per_layer.values() if v['binary']]
    if rates and any(r > 0 for r in rates.values()):
        mean_fr = sum(rates.values()) / len(rates)
        fr_source = 'spiking_neuron_modules'
    elif binary_rates:
        mean_fr = sum(binary_rates) / len(binary_rates)
        fr_source = 'binary_layer_inputs'
    else:
        mean_fr = 0.0
        fr_source = 'none_detected'

    # Dense reference: prefer the dispatch-level count (which sees functional
    # matmuls) so the FLOPs column and the ANN energy come from one source.
    if dense_gflops is not None:
        total_ann_flops = dense_gflops * 1e9
    e_ann = total_ann_flops * e_mac
    return {
        'energy_accounting_version': ENERGY_ACCOUNTING_VERSION,
        # primary estimate and the optimistic lower bound (multibit-as-AC)
        'E_primary_uJ': e_snn * 1e6,
        'E_optimistic_lower_uJ': e_opt * 1e6,
        'binary_ac_uJ': binary_ac_J * 1e6,
        'multibit_mac_uJ': multibit_mac_J * 1e6,
        'static_input_mac_uJ': static_input_mac_J * 1e6,
        'classifier_uJ': classifier_J * 1e6,
        'classifier_input_kinds': {k: v['input_kind'] for k, v in per_layer.items()
                                   if v['component'] == 'classifier'},
        'E_SNN_uJ': e_snn * 1e6,
        'E_ANN_uJ': e_ann * 1e6,
        'energy_ratio': e_snn / max(e_ann, 1e-30),
        'SOPs_M': total_sops / 1e6,
        'MAC_ops_M': total_mac_ops / 1e6,
        'ANN_GFLOPs': total_ann_flops / 1e9,
        'mean_firing_rate': mean_fr,
        'firing_rate_source': fr_source,
        'spike_rates': rates,
        # Spike evidence for models whose neurons are functional calls rather
        # than Modules (no hook can see them): the measured 0/1 input rate of
        # every weighted layer that consumes spikes.
        'binary_input_rates': {k: v['rate'] for k, v in per_layer.items()
                               if v['binary']},
        'n_profile_samples': n_samples,
        'lif_rates': rates,
        'per_layer': per_layer,
        'components': {k: {'SOPs_M': v['SOPs'] / 1e6, 'MACs_M': v['MACs'] / 1e6,
                           'energy_uJ': v['energy_J'] * 1e6}
                       for k, v in components.items()},
    }


def _attention_ops(model, rates, T):
    """SOPs and ANN-FLOPs of the functional matmuls inside attention blocks."""
    from .models.attention import AdditiveSpikingAttention, SpikeDrivenAttention

    sops, ann_flops = 0.0, 0.0
    N = getattr(model, 'num_tokens', None)
    if N is None and hasattr(model, 'pos_embed'):
        N = model.pos_embed.shape[-2]
    if N is None:
        return 0.0, 0.0

    for name, m in model.named_modules():
        if isinstance(m, (AdditiveSpikingAttention, SpikeDrivenAttention)):
            H, d = m.qkv.num_heads, m.qkv.head_dim
            r = lambda suffix: rates.get(f'{name}.{suffix}', 0.0)
            if isinstance(m, AdditiveSpikingAttention):
                # (q*w) @ k^T driven by q spikes; attn @ v driven by attn spikes
                sops += T * H * N * N * d * (r('qkv.q_lif') + r('attn_lif'))
                ann_flops += 2 * H * N * N * d
            elif isinstance(m, SpikeDrivenAttention):
                # (k AND v) accumulate + q Hadamard mask. Use the measured joint
                # co-activity when available; fall back to the independence
                # approximation (product of marginals) only if never profiled.
                kv_rate = m.kv_rate
                if kv_rate is None:
                    kv_rate = r('qkv.k_lif') * r('qkv.v_lif')
                sops += T * H * N * d * kv_rate + T * N * H * d * r('qkv.q_lif')
                ann_flops += 2 * N * H * d
    return sops, ann_flops


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def ann_gflops(model, input_shape, device):
    """Dense (full-precision) GFLOPs for one patch inference.

    Uses torch's dispatch-level FLOP counter so that *functional* operations are
    included -- scaled-dot-product attention inside the transformer baselines and
    the `h @ A` state update of the recurrent SSM are invisible to module hooks and
    were previously undercounted. FLOPs are weight-independent, so this needs no
    trained checkpoint. Reported per timestep-equivalent (divided by T) so spiking
    and non-spiking models are compared on the same one-step basis.
    """
    from torch.utils.flop_counter import FlopCounterMode

    x = torch.randn(1, *input_shape, device=device)
    # PyTorch's fused transformer fast path (taken only in eval) hides the
    # attention batched matmuls from the dispatcher, which silently undercounts
    # the transformer baselines. Put *only* the transformer modules in train mode
    # to disable that path -- the arithmetic is identical, no weights change under
    # no_grad, and normalization layers stay in eval so batch-1 input is valid.
    model.eval()
    switched = [m for m in model.modules()
                if isinstance(m, (nn.TransformerEncoder, nn.TransformerEncoderLayer,
                                  nn.MultiheadAttention))]
    for m in switched:
        m.train()
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    functional.reset_net(model)
    for m in switched:
        m.eval()
    # FlopCounterMode reports 2*MACs for matmul/conv; our E_MAC accounting is
    # per multiply-accumulate, so halve it to stay in MAC units.
    macs = counter.get_total_flops() / 2.0
    T = getattr(model, 'timesteps', 1)
    return macs / T / 1e9
