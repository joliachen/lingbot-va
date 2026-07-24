# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Video ICL for LingBot-MoT: inference-time demo KV injection (planning_icl.md).

Mechanism
---------
The inference KV cache is a per-block slot pool (WanAttention.attn_caches);
every self-attention forward attends ALL valid slots. We keep the demo KV
OUTSIDE that pool and concatenate `demo_kv[:p_hat]` into the key/value set of
self-attention forwards ONLY while ICL is enabled. The ICL server enables the
state during `_infer` (prediction/denoise forwards) and disables it during
`_compute_kv_cache` (history commits), which realizes the plan's mask rule
with zero approximation:

    - prediction tokens see:  demo[:p_hat] | history pool | current chunk
    - history tokens never see demo -> history KV identical to the baseline,
      reusable and exact (E0a comparability).

Positions (RoPE)
----------------
Demo KV is encoded offline at native latent-frame positions [0, T_demo) with
the same per-chunk causal procedure the server uses for real history. At
injection time the sliced prefix K is re-rotated by a uniform frame delta

    delta = -(gap_frames + p_hat_frames)

so the injected demo occupies t in [-gap - p_hat, -gap), strictly before the
rollout history (which starts at t=0), per planning_icl.md section 2.3. Only
K is rotated (V carries no positional encoding). Rotation touches only the
temporal frequency dims (first f_dim//2 complex dims of head_dim/2).

Nothing in model.py / model_mot.py is modified: we monkey-patch the bound
`forward` of each block's attn1 / action_attn1 instance on an already-loaded
model. The patched forward is byte-for-byte the original WanAttention.forward
plus the demo concat + optional attention-mass diagnostic.
"""
import types

import torch

__all__ = ['ICLState', 'install_icl', 'export_demo_kv', 'rotate_k_frames']


def _f_dim(head_dim: int) -> int:
    # matches WanRotaryPosEmbed: f_dim = head_dim - 2 * (head_dim // 3)
    return head_dim - 2 * (head_dim // 3)


def rotate_k_frames(k: torch.Tensor, delta_frames: float, theta: float = 10000.0):
    """Shift the temporal RoPE position of post-RoPE keys by `delta_frames`.

    k: [B, T, H, D] (post-RoPE). Returns a new tensor of the same dtype.
    Only the first f_dim//2 complex dims (temporal freqs) are rotated;
    height/width dims are position-identical between demo and rollout.
    """
    if delta_frames == 0:
        return k
    head_dim = k.shape[-1]
    fd = _f_dim(head_dim)
    f_freqs_base = 1.0 / (theta ** (torch.arange(
        0, fd, 2, device=k.device)[:(fd // 2)].double() / fd))
    phase = torch.polar(torch.ones_like(f_freqs_base),
                        f_freqs_base * float(delta_frames))  # complex128 [fd//2]
    k_c = torch.view_as_complex(
        k.to(torch.float64).reshape(*k.shape[:-1], -1, 2))  # [B,T,H,D/2] c128
    k_c[..., :fd // 2] = k_c[..., :fd // 2] * phase
    return torch.view_as_real(k_c).flatten(-2).to(k.dtype)


class ICLState:
    """Shared mutable state read by all patched attention forwards."""

    def __init__(self, demo_k, demo_v, tokens_per_chunk, n_chunks,
                 gap_frames=4, frames_per_chunk=4, capture_attn_mass=False):
        # demo_k/demo_v: [n_layers][B, N_tokens, H, D] on device, native positions
        self.demo_k = demo_k
        self.demo_v = demo_v
        self.tokens_per_chunk = tokens_per_chunk
        self.n_chunks = n_chunks
        self.gap_frames = gap_frames
        self.frames_per_chunk = frames_per_chunk
        self.enabled = False          # toggled by the ICL server around _infer
        self.p_hat = 0                # chunks currently visible (after lookahead)
        self._demo_k_rot = None       # rotated+sliced K per layer for current p_hat
        self.capture_attn_mass = capture_attn_mass
        # filled per captured forward: {'video'|'action'}[layer] = mass scalar
        self.attn_mass = {'video': {}, 'action': {}}

    @property
    def active_tokens(self):
        return self.p_hat * self.tokens_per_chunk

    def set_progress(self, p_hat_chunks: int):
        """Slice demo[:p_hat] and re-rotate K so the slice ends at -gap."""
        p = max(0, min(int(p_hat_chunks), self.n_chunks))
        self.p_hat = p
        if p == 0:
            self._demo_k_rot = None
            return
        n_tok = p * self.tokens_per_chunk
        delta = -(self.gap_frames + p * self.frames_per_chunk)
        self._demo_k_rot = [
            rotate_k_frames(k[:, :n_tok], delta) for k in self.demo_k
        ]

    def kv_for_layer(self, layer_idx, batch_size):
        """Return (k, v) demo slices for this layer, batch-matched, or None."""
        if not self.enabled or self.p_hat == 0:
            return None
        k = self._demo_k_rot[layer_idx]
        v = self.demo_v[layer_idx][:, :self.active_tokens]
        if k.shape[0] != batch_size:
            if k.shape[0] == 1:
                k = k.expand(batch_size, -1, -1, -1)
                v = v.expand(batch_size, -1, -1, -1)
            else:  # stored with CFG batch 2 [cond, uncond], rollout without CFG
                k = k[:batch_size]
                v = v[:batch_size]
        return k, v


def _make_icl_forward(state, layer_idx, stream):
    """Build a replacement for WanAttention.forward with demo-KV concat."""

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name='pos'):
        kv_cache = self.attn_caches[cache_name] if (
            self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query).unflatten(2, (self.heads, -1))
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:
            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            slots = self.update_cache(cache_name, key, value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        # -- ICL: prepend demo[:p_hat] KV (never written into the pool) --
        demo = state.kv_for_layer(layer_idx, key.shape[0])
        if demo is not None:
            demo_k, demo_v = demo
            n_demo = demo_k.shape[1]
            key = torch.cat([demo_k.to(key.dtype), key], dim=1)
            value = torch.cat([demo_v.to(value.dtype), value], dim=1)
            if state.capture_attn_mass and update_cache == 1:
                with torch.no_grad():
                    # cond row only; fp32 softmax over the full key set
                    qf = query[0:1].float().permute(0, 2, 1, 3)  # 1,H,L,D
                    kf = key[0:1].float().permute(0, 2, 1, 3)
                    logits = qf @ kf.transpose(-1, -2) / (qf.shape[-1] ** 0.5)
                    attn = logits.softmax(dim=-1)
                    mass = attn[..., :n_demo].sum(-1).mean().item()
                state.attn_mass[stream][layer_idx] = mass

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states

    return forward


def install_icl(transformer, state):
    """Patch attn1/action_attn1 of every block on a loaded MoT model.

    Only self-attention instances are patched (cross-attn has no pool and no
    demo). Patching replaces the bound method on the *instance*; model code
    files stay untouched and other model instances are unaffected.
    """
    for layer_idx, block in enumerate(transformer.blocks):
        block.attn1.forward = types.MethodType(
            _make_icl_forward(state, layer_idx, 'video'), block.attn1)
        if hasattr(block, 'action_attn1'):
            block.action_attn1.forward = types.MethodType(
                _make_icl_forward(state, layer_idx, 'action'), block.action_attn1)
    return transformer


@torch.no_grad()
def export_demo_kv(transformer, cache_name='demo'):
    """Export per-layer post-RoPE K/V from a fully-encoded demo cache pool.

    Requires the pool to have been sized exactly (no eviction): slots are then
    allocated sequentially, so pool[:, :n_valid] is in temporal order. Returns
    (k_list, v_list) with tensors [B, N, H, D] on CPU.
    """
    k_list, v_list = [], []
    for block in transformer.blocks:
        cache = block.attn1.attn_caches[cache_name]
        mask = cache['mask']
        n_valid = int(mask.sum().item())
        assert bool(mask[:n_valid].all()) and not bool(mask[n_valid:].any()), (
            'demo cache pool has holes -- was it sized exactly (no eviction)?')
        ids = cache['id'][:n_valid]
        assert bool((ids[1:] >= ids[:-1]).all()), 'demo cache ids not ascending'
        k_list.append(cache['k'][:, :n_valid].detach().cpu().clone())
        v_list.append(cache['v'][:, :n_valid].detach().cpu().clone())
    return k_list, v_list
