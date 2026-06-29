# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import warnings
"""
Mixture-of-Transformers (MoT) model for LingBot-VA.

Paper Section 3.3 / 4.2:
  - Video stream:  d_v=3072, 24 heads × 128 head_dim  (WAN2.2-5B backbone)
  - Action stream: d_a=768,  12 heads × 64  head_dim  (~350M additional params)
  - At every layer: action tokens projected d_a→d_v, joint self-attention
    with separate QKV projection matrices, then projected back d_v→d_a + residual
  - Separate text cross-attention and FFN per stream
  - Weight init: action weights interpolated from video × α = √(d_v/d_a) = 2

Training:  full cross-modal joint self-attention (FlexAttn mask reused—same token count)
Inference: video and action passes run separately;
           action blocks attend to per-block video K/V cached from the most
           recent video denoising pass (cross-modal grounding)
"""
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

from .model import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    FlexAttnFunc,
    flash_attn_func,
    custom_sdpa,
)

__all__ = ['WanMoTTransformer3DModel', 'init_mot_from_video_weights']

# ── Paper Table / Section 4.2 ─────────────────────────────────────────────────
_D_V        = 3072
_D_A        = 768
_N_HEADS_V  = 24
_N_HEADS_A  = 12
_HEAD_DIM_V = _D_V // _N_HEADS_V   # 128
_HEAD_DIM_A = _D_A // _N_HEADS_A   # 64
_FFN_DIM_A  = _D_A * 4             # 3072
_ALPHA      = math.sqrt(_D_V / _D_A)  # 2.0 — weight init scale


# ─────────────────────────────────────────────────────────────────────────────
def _apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x64 = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(x64 * freqs).flatten(-2).to(x.dtype)


# ─────────────────────────────────────────────────────────────────────────────
class WanMoTBlock(nn.Module):
    """
    One MoT layer.

    Self-attention (§3.3 paper-correct order):
      norm_v, norm_a  (separate adaLN)
      video  QKV: W_q/k/v_v in d_v space  → 24 heads × 128
      action QKV: W_q/k/v_a in native d_a space (12 heads × 64),
                  then per-QKV up-projected d_a→d_v via action_{q,k,v}_up
      RoPE applied to both video Q/K and action Q/K (§4.2: "Both streams employ RoPE")
      joint flash-attn on [video ‖ action tokens all at d_v]
      → split → video_to_out (d_v) / action_down (d_v→d_a) + residual

    Text cross-attention:
      video:  Q ∈ d_v, K/V ∈ d_v  (text already in d_v from text_embedder)
      action: Q ∈ d_a, K/V ∈ d_a  (text projected down via action_text_proj)

    FFN:
      video FFN  (d_v → ffn_v → d_v)
      action FFN (d_a → ffn_a → d_a)

    Inference cross-modal:
      forward_video stores per-block K/V in _cross_video_k/v.
      forward_action prepends those K/V so action can attend to video context.
    """

    def __init__(
        self,
        d_v: int,
        d_a: int,
        ffn_dim_v: int,
        ffn_dim_a: int,
        num_heads_v: int,
        num_heads_a: int,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        attn_mode: str = 'torch',
    ):
        super().__init__()
        head_dim_v = d_v // num_heads_v
        head_dim_a = d_a // num_heads_a

        # ── video stream ──────────────────────────────────────────────────
        self.video_norm1 = FP32LayerNorm(d_v, eps, elementwise_affine=False)
        self.video_scale_shift_table = nn.Parameter(
            torch.randn(1, 6, d_v) / d_v ** 0.5)

        self.video_to_q   = nn.Linear(d_v, d_v, bias=True)
        self.video_to_k   = nn.Linear(d_v, d_v, bias=True)
        self.video_to_v   = nn.Linear(d_v, d_v, bias=True)
        self.video_norm_q = nn.RMSNorm(d_v, eps=eps)
        self.video_norm_k = nn.RMSNorm(d_v, eps=eps)
        self.video_to_out = nn.Linear(d_v, d_v, bias=True)

        self.video_cross_attn = WanAttention(
            dim=d_v, heads=num_heads_v, dim_head=head_dim_v,
            eps=eps, cross_attention_dim_head=head_dim_v, attn_mode=attn_mode)
        self.video_norm2 = (FP32LayerNorm(d_v, eps, elementwise_affine=True)
                            if cross_attn_norm else nn.Identity())
        self.video_ffn   = FeedForward(d_v, inner_dim=ffn_dim_v,
                                       activation_fn='gelu-approximate')
        self.video_norm3 = FP32LayerNorm(d_v, eps, elementwise_affine=False)

        # ── action stream ─────────────────────────────────────────────────
        self.action_norm1 = FP32LayerNorm(d_a, eps, elementwise_affine=False)
        self.action_scale_shift_table = nn.Parameter(
            torch.randn(1, 6, d_a) / d_a ** 0.5)

        # action QKV in native d_a space (§3.3: "maintaining distinct feature spaces")
        # 12 heads × 64 head_dim
        self.action_to_q   = nn.Linear(d_a, d_a, bias=True)
        self.action_to_k   = nn.Linear(d_a, d_a, bias=True)
        self.action_to_v   = nn.Linear(d_a, d_a, bias=True)
        self.action_norm_q = nn.RMSNorm(d_a, eps=eps)
        self.action_norm_k = nn.RMSNorm(d_a, eps=eps)

        # per-QKV up-projections d_a → d_v for joint self-attention
        # (§3.3: "projected to video dimension via a linear layer")
        self.action_q_up = nn.Linear(d_a, d_v, bias=True)
        self.action_k_up = nn.Linear(d_a, d_v, bias=True)
        self.action_v_up = nn.Linear(d_a, d_v, bias=True)

        # output down-projection d_v → d_a after joint attention
        self.action_down = nn.Linear(d_v, d_a, bias=True)

        # action text cross-attention in d_a space
        self.action_text_proj = nn.Linear(d_v, d_a, bias=True)
        self.action_cross_attn = WanAttention(
            dim=d_a, heads=num_heads_a, dim_head=head_dim_a,
            eps=eps, cross_attention_dim_head=head_dim_a, attn_mode=attn_mode)
        self.action_norm2 = (FP32LayerNorm(d_a, eps, elementwise_affine=True)
                             if cross_attn_norm else nn.Identity())
        self.action_ffn   = FeedForward(d_a, inner_dim=ffn_dim_a,
                                        activation_fn='gelu-approximate')
        self.action_norm3 = FP32LayerNorm(d_a, eps, elementwise_affine=False)

        # inference KV caches (populated by forward_video / forward_action)
        self._video_kv:  dict = {}   # cache_name → (K, V) tensors
        self._action_kv: dict = {}
        self._cross_video_k: torch.Tensor | None = None   # from last video pass
        self._cross_video_v: torch.Tensor | None = None

        self.num_heads_v = num_heads_v
        self.num_heads_a = num_heads_a
        self.head_dim_v  = head_dim_v
        self.d_v = d_v
        self.d_a = d_a
        self.attn_mode   = attn_mode

    # ── helpers ───────────────────────────────────────────────────────────
    def _heads_v(self, x):
        return x.unflatten(2, (self.num_heads_v, self.head_dim_v))

    def _raw_attn(self, q, k, v):
        if self.attn_mode == 'flashattn':
            return flash_attn_func(q, k, v)
        elif self.attn_mode == 'flex':
            return FlexAttnFunc(is_cross=False)(q, k, v)
        return custom_sdpa(q, k, v)

    def _ada_ln_v(self, x, temb):
        tbl = self.video_scale_shift_table[None] + temb.float()
        parts = rearrange(tbl, 'b l n c -> b n l c').chunk(6, dim=1)
        shift, scale, gate, c_shift, c_scale, c_gate = (p.squeeze(1) for p in parts)
        return (self.video_norm1(x.float()) * (1. + scale) + shift).type_as(x), \
               gate, c_shift, c_scale, c_gate

    def _ada_ln_a(self, x, temb):
        tbl = self.action_scale_shift_table[None] + temb.float()
        parts = rearrange(tbl, 'b l n c -> b n l c').chunk(6, dim=1)
        shift, scale, gate, c_shift, c_scale, c_gate = (p.squeeze(1) for p in parts)
        return (self.action_norm1(x.float()) * (1. + scale) + shift).type_as(x), \
               gate, c_shift, c_scale, c_gate

    # ── training: joint cross-modal forward ───────────────────────────────
    def forward_joint(
        self,
        video_states:  torch.Tensor,         # (1, L_v, d_v)
        action_states: torch.Tensor,         # (1, L_a, d_a)
        encoder_hidden_states: torch.Tensor, # (1, L_t, d_v)
        video_temb:   torch.Tensor,          # (1, L_v, 6, d_v)
        action_temb:  torch.Tensor,          # (1, L_a, 6, d_a)
        rotary_emb:   torch.Tensor,          # (1, L_v+L_a, 1, head_dim_v)
    ):
        L_v = video_states.shape[1]
        L_a = action_states.shape[1]

        norm_v, v_gate, v_c_shift, v_c_scale, v_c_gate = self._ada_ln_v(video_states, video_temb)
        norm_a, a_gate, a_c_shift, a_c_scale, a_c_gate = self._ada_ln_a(action_states, action_temb)

        # video QKV in d_v space (24 heads × 128)
        Q_v = self._heads_v(self.video_norm_q(self.video_to_q(norm_v)))
        K_v = self._heads_v(self.video_norm_k(self.video_to_k(norm_v)))
        V_v = self._heads_v(self.video_to_v(norm_v))

        # action QKV in native d_a space (§3.3), then up-projected to d_v for joint SA
        Q_a_nat = self.action_norm_q(self.action_to_q(norm_a))   # (1, L_a, d_a)
        K_a_nat = self.action_norm_k(self.action_to_k(norm_a))   # (1, L_a, d_a)
        V_a_nat = self.action_to_v(norm_a)                        # (1, L_a, d_a)
        Q_a = self._heads_v(self.action_q_up(Q_a_nat))            # (1, L_a, 24, 128)
        K_a = self._heads_v(self.action_k_up(K_a_nat))
        V_a = self._heads_v(self.action_v_up(V_a_nat))

        # RoPE: §4.2 "Both streams employ RoPE positional encoding"
        # Video uses 3D (t,h,w) grid; action uses 1D temporal grid (h=w=0)
        Q_v = _apply_rotary(Q_v, rotary_emb[:, :L_v])
        K_v = _apply_rotary(K_v, rotary_emb[:, :L_v])
        Q_a = _apply_rotary(Q_a, rotary_emb[:, L_v:L_v + L_a])
        K_a = _apply_rotary(K_a, rotary_emb[:, L_v:L_v + L_a])

        # joint self-attention [video ‖ action_up]
        Q = torch.cat([Q_v, Q_a], dim=1)
        K = torch.cat([K_v, K_a], dim=1)
        V = torch.cat([V_v, V_a], dim=1)
        out = self._raw_attn(Q, K, V).flatten(2, 3)   # (1, L_v+L_a, d_v)

        # split and project back
        v_out = self.video_to_out(out[:, :L_v])        # (1, L_v, d_v)
        a_out = self.action_down(out[:, L_v:])          # (1, L_a, d_a)

        video_states  = (video_states.float()  + v_out * v_gate).type_as(video_states)
        action_states = (action_states.float() + a_out * a_gate).type_as(action_states)

        # text cross-attention — video (d_v) and action (d_a separately)
        norm_v2 = self.video_norm2(video_states.float()).type_as(video_states)
        video_states = video_states + self.video_cross_attn(
            norm_v2, encoder_hidden_states, encoder_hidden_states, None)

        text_da = self.action_text_proj(encoder_hidden_states)  # (1, L_t, d_a)
        norm_a2 = self.action_norm2(action_states.float()).type_as(action_states)
        action_states = action_states + self.action_cross_attn(
            norm_a2, text_da, text_da, None)

        # FFN — separate per stream
        norm_v3 = (self.video_norm3(video_states.float()) *
                   (1. + v_c_scale) + v_c_shift).type_as(video_states)
        video_states = (video_states.float() +
                        self.video_ffn(norm_v3).float() * v_c_gate).type_as(video_states)

        norm_a3 = (self.action_norm3(action_states.float()) *
                   (1. + a_c_scale) + a_c_shift).type_as(action_states)
        action_states = (action_states.float() +
                         self.action_ffn(norm_a3).float() * a_c_gate).type_as(action_states)

        return video_states, action_states

    # ── inference: video stream only ──────────────────────────────────────
    def forward_video(
        self,
        video_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        video_temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        update_cache: int = 0,
        cache_name: str = 'pos',
    ) -> torch.Tensor:
        L_v = video_states.shape[1]
        norm_v, v_gate, v_c_shift, v_c_scale, v_c_gate = self._ada_ln_v(video_states, video_temb)

        Q = self._heads_v(self.video_norm_q(self.video_to_q(norm_v)))
        K = self._heads_v(self.video_norm_k(self.video_to_k(norm_v)))
        V = self._heads_v(self.video_to_v(norm_v))

        Q = _apply_rotary(Q, rotary_emb[:, :L_v])
        K = _apply_rotary(K, rotary_emb[:, :L_v])

        # accumulate KV cache
        cached = self._video_kv.get(cache_name)
        if cached is not None:
            K_full = torch.cat([cached[0], K], dim=1)
            V_full = torch.cat([cached[1], V], dim=1)
        else:
            K_full, V_full = K, V

        if update_cache != 0:
            self._video_kv[cache_name] = (K_full.detach(), V_full.detach())

        out = self._raw_attn(Q, K_full, V_full).flatten(2, 3)
        v_out = self.video_to_out(out)

        video_states = (video_states.float() + v_out * v_gate).type_as(video_states)

        # store cross-modal K/V for action to use
        self._cross_video_k = K.detach()
        self._cross_video_v = V.detach()

        norm_v2 = self.video_norm2(video_states.float()).type_as(video_states)
        video_states = video_states + self.video_cross_attn(
            norm_v2, encoder_hidden_states, encoder_hidden_states, None)

        norm_v3 = (self.video_norm3(video_states.float()) *
                   (1. + v_c_scale) + v_c_shift).type_as(video_states)
        video_states = (video_states.float() +
                        self.video_ffn(norm_v3).float() * v_c_gate).type_as(video_states)
        return video_states

    # ── inference: action stream only ─────────────────────────────────────
    def forward_action(
        self,
        action_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        action_temb: torch.Tensor,
        rotary_emb: torch.Tensor | None = None,
        update_cache: int = 0,
        cache_name: str = 'pos',
    ) -> torch.Tensor:
        norm_a, a_gate, a_c_shift, a_c_scale, a_c_gate = self._ada_ln_a(action_states, action_temb)

        Q_a_nat = self.action_norm_q(self.action_to_q(norm_a))
        K_a_nat = self.action_norm_k(self.action_to_k(norm_a))
        V_a_nat = self.action_to_v(norm_a)
        Q = self._heads_v(self.action_q_up(Q_a_nat))
        K = self._heads_v(self.action_k_up(K_a_nat))
        V = self._heads_v(self.action_v_up(V_a_nat))

        if rotary_emb is not None:
            Q = _apply_rotary(Q, rotary_emb)
            K = _apply_rotary(K, rotary_emb)

        # build keys/values: action history + cross-modal video
        k_parts, v_parts = [], []
        cached = self._action_kv.get(cache_name)
        if cached is not None:
            k_parts.append(cached[0])
            v_parts.append(cached[1])
        k_parts.append(K)
        v_parts.append(V)
        if self._cross_video_k is not None:
            k_parts.append(self._cross_video_k.to(K.device))
            v_parts.append(self._cross_video_v.to(V.device))

        K_full = torch.cat(k_parts, dim=1)
        V_full = torch.cat(v_parts, dim=1)

        if update_cache != 0:
            new_k = K if cached is None else torch.cat([cached[0], K], dim=1)
            new_v = V if cached is None else torch.cat([cached[1], V], dim=1)
            self._action_kv[cache_name] = (new_k.detach(), new_v.detach())

        out_dv = self._raw_attn(Q, K_full, V_full).flatten(2, 3)
        a_out  = self.action_down(out_dv)
        action_states = (action_states.float() + a_out * a_gate).type_as(action_states)

        text_da = self.action_text_proj(encoder_hidden_states)
        norm_a2 = self.action_norm2(action_states.float()).type_as(action_states)
        action_states = action_states + self.action_cross_attn(
            norm_a2, text_da, text_da, None)

        norm_a3 = (self.action_norm3(action_states.float()) *
                   (1. + a_c_scale) + a_c_shift).type_as(action_states)
        action_states = (action_states.float() +
                         self.action_ffn(norm_a3).float() * a_c_gate).type_as(action_states)
        return action_states

    # ── cache management ──────────────────────────────────────────────────
    def clear_video_cache(self, name):
        self._video_kv[name] = None

    def clear_action_cache(self, name):
        self._action_kv[name] = None

    def clear_cross_modal(self):
        self._cross_video_k = None
        self._cross_video_v = None


# ─────────────────────────────────────────────────────────────────────────────
class WanMoTTransformer3DModel(ModelMixin, ConfigMixin):
    """
    Drop-in replacement for WanTransformer3DModel using MoT architecture.

    New hyperparameters vs shared backbone:
      action_inner_dim   = 768  (d_a)
      action_num_heads   = 12
      action_ffn_dim     = 3072

    The public forward() / forward_train() signatures are identical to the
    shared-backbone model so that VA_Server requires no changes.
    """
    _supports_gradient_checkpointing = True
    _no_split_modules = ['WanMoTBlock']
    _skip_layerwise_casting_patterns = ['patch_embedding_mlp', 'condition_embedder', 'norm']
    _keep_in_fp32_modules = [
        'time_embedder', 'scale_shift_table', 'action_scale_shift_table',
        'video_norm1', 'video_norm3', 'action_norm1', 'action_norm3',
    ]

    @register_to_config
    def __init__(
        self,
        patch_size           = [1, 2, 2],
        num_attention_heads  = 24,
        attention_head_dim   = 128,
        in_channels          = 48,
        out_channels         = 48,
        action_dim           = 30,
        text_dim             = 4096,
        freq_dim             = 256,
        ffn_dim              = 14336,
        num_layers           = 30,
        cross_attn_norm      = True,
        eps                  = 1e-6,
        rope_max_seq_len     = 1024,
        pos_embed_seq_len    = None,
        attn_mode            = 'torch',
        # MoT-specific
        action_inner_dim     = _D_A,
        action_num_heads     = _N_HEADS_A,
        action_ffn_dim       = _FFN_DIM_A,
    ):
        super().__init__()
        self.patch_size          = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim  = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim   # d_v = 3072

        # ── video embedding ───────────────────────────────────────────────
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6, text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len)

        # ── action embedding (smaller, d_a) ───────────────────────────────
        self.action_embedder = nn.Linear(action_dim, action_inner_dim)
        # adaLN for action stream — outputs d_a × 6 per token
        self.condition_embedder_action = WanTimeTextImageEmbedding(
            dim=action_inner_dim, time_freq_dim=freq_dim,
            time_proj_dim=action_inner_dim * 6, text_embed_dim=text_dim,
            pos_embed_seq_len=None)

        # ── MoT blocks ───────────────────────────────────────────────────
        self.mot_blocks = nn.ModuleList([
            WanMoTBlock(
                d_v=inner_dim, d_a=action_inner_dim,
                ffn_dim_v=ffn_dim, ffn_dim_a=action_ffn_dim,
                num_heads_v=num_attention_heads, num_heads_a=action_num_heads,
                cross_attn_norm=cross_attn_norm, eps=eps, attn_mode=attn_mode,
            )
            for _ in range(num_layers)
        ])

        # ── video output head ─────────────────────────────────────────────
        self.norm_out     = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out     = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim ** 0.5)

        # ── action output head (d_a) ──────────────────────────────────────
        self.action_norm_out = FP32LayerNorm(action_inner_dim, eps, elementwise_affine=False)
        self.action_proj_out = nn.Linear(action_inner_dim, action_dim)
        self.action_scale_shift_table = nn.Parameter(
            torch.randn(1, 2, action_inner_dim) / action_inner_dim ** 0.5)

        self._action_inner_dim = action_inner_dim

    # ── helpers ───────────────────────────────────────────────────────────
    def _video_embed(self, latents):
        x = rearrange(latents,
                      'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                      p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
        return self.patch_embedding_mlp(x)

    def _action_embed(self, actions):
        x = rearrange(actions, 'b c f h w -> b (f h w) c')
        return self.action_embedder(x)

    def _time_embed_video(self, timesteps, H, W, dtype):
        ts = torch.repeat_interleave(
            timesteps,
            (H // self.patch_size[1]) * (W // self.patch_size[2]), dim=1)
        temb, tproj = self.condition_embedder(ts, dtype=dtype)
        return temb, tproj.unflatten(2, (6, -1))

    def _time_embed_action(self, timesteps, H, W, dtype):
        # action tokens have patch_size=(1,1,1)
        ts = torch.repeat_interleave(timesteps, H * W, dim=1)
        temb, tproj = self.condition_embedder_action(ts, dtype=dtype)
        return temb, tproj.unflatten(2, (6, -1))

    # ── cache management ──────────────────────────────────────────────────
    def clear_cache(self, cache_name):
        for b in self.mot_blocks:
            b.clear_video_cache(cache_name)
            b.clear_action_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        # simplified: just clear whole cache (no slot-level pred tracking yet)
        self.clear_cache(cache_name)

    def create_empty_cache(self, cache_name, *args, **kwargs):
        # MoT uses dynamic list-based cache; no pre-allocation needed
        for b in self.mot_blocks:
            b._video_kv[cache_name]  = None
            b._action_kv[cache_name] = None

    # ── training forward ──────────────────────────────────────────────────
    def forward_train(self, input_dict):
        ld = input_dict['latent_dict']
        ad = input_dict['action_dict']
        batch_size = ld['noisy_latents'].shape[0]

        # cast to bf16
        for d in (ld, ad):
            d['noisy_latents'] = d['noisy_latents'].to(torch.bfloat16)
            d['latent']        = d['latent'].to(torch.bfloat16)

        # embed
        v_noise = self._video_embed(ld['noisy_latents']).flatten(0, 1)[None]   # (1, B*L_v, d_v)
        v_cond  = self._video_embed(ld['latent']).flatten(0, 1)[None]
        a_noise = self._action_embed(ad['noisy_latents']).flatten(0, 1)[None]  # (1, B*L_a, d_a)
        a_cond  = self._action_embed(ad['latent']).flatten(0, 1)[None]

        text_hs = self.condition_embedder.text_embedder(
            ld['text_emb']).flatten(0, 1)[None]                                # (1, B*L_t, d_v)

        # rotary (video only)
        v_grid = ld['grid_id'].permute(1, 0, 2).flatten(1)[None]
        a_grid = ad['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid = torch.cat([v_grid] * 2 + [a_grid] * 2, dim=2)
        rotary_emb = self.rope(full_grid)[:, :, None]

        # timestep embeddings (video: d_v, action: d_a)
        v_ts_all = torch.cat([ld['timesteps'].flatten(0,1), ld['cond_timesteps'].flatten(0,1)])[None]
        a_ts_all = torch.cat([ad['timesteps'].flatten(0,1), ad['cond_timesteps'].flatten(0,1)])[None]

        _, v_tproj = self._time_embed_video(
            v_ts_all, ld['noisy_latents'].shape[-2], ld['noisy_latents'].shape[-1],
            dtype=torch.bfloat16)
        _, a_tproj = self._time_embed_action(
            a_ts_all, ad['noisy_latents'].shape[-2], ad['noisy_latents'].shape[-1],
            dtype=torch.bfloat16)
        v_temb_all = torch.cat([v_tproj], dim=1)   # already (1, 2*L_v, 6, d_v)
        a_temb_all = torch.cat([a_tproj], dim=1)

        # concatenate noisy+cond for each stream
        video_states  = torch.cat([v_noise, v_cond], dim=1)   # (1, 2*L_v, d_v)
        action_states = torch.cat([a_noise, a_cond], dim=1)   # (1, 2*L_a, d_a)

        # Init FlexAttn mask (same token-count as shared backbone — mask is position-based)
        FlexAttnFunc.init_mask(
            ld['noisy_latents'].shape, ad['noisy_latents'].shape,
            padded_length=0,
            chunk_size=input_dict['chunk_size'],
            window_size=input_dict['window_size'],
            patch_size=self.patch_size,
            device=video_states.device)

        for block in self.mot_blocks:
            video_states, action_states = block.forward_joint(
                video_states, action_states, text_hs,
                v_temb_all, a_temb_all, rotary_emb)

        L_v_single = v_noise.shape[1]
        L_a_single = a_noise.shape[1]

        # video output norm + unpatch
        v_temb_out, _ = self._time_embed_video(
            v_ts_all[:, :v_noise.shape[1]], ld['noisy_latents'].shape[-2],
            ld['noisy_latents'].shape[-1], dtype=torch.bfloat16)
        v_tbl = self.scale_shift_table[None] + v_temb_out[:, :L_v_single, None, :]
        shift_v, scale_v = rearrange(v_tbl, 'b l n c -> b n l c').chunk(2, dim=1)
        v_out = video_states[:, :L_v_single]
        v_out = (self.norm_out(v_out.float()) * (1. + scale_v.squeeze(1)) +
                 shift_v.squeeze(1)).type_as(v_out)
        v_out = self.proj_out(v_out)
        v_out = rearrange(v_out, '1 (b l) (n c) -> b (l n) c',
                          n=math.prod(self.patch_size), b=batch_size)

        # action output norm
        a_temb_out, _ = self._time_embed_action(
            a_ts_all[:, :a_noise.shape[1]], ad['noisy_latents'].shape[-2],
            ad['noisy_latents'].shape[-1], dtype=torch.bfloat16)
        a_tbl = self.action_scale_shift_table[None] + a_temb_out[:, :L_a_single, None, :]
        shift_a, scale_a = rearrange(a_tbl, 'b l n c -> b n l c').chunk(2, dim=1)
        a_out = action_states[:, :L_a_single]
        a_out = (self.action_norm_out(a_out.float()) * (1. + scale_a.squeeze(1)) +
                 shift_a.squeeze(1)).type_as(a_out)
        a_out = self.action_proj_out(a_out)
        a_out = rearrange(a_out, '1 (b l) c -> b l c', b=batch_size)

        return v_out, a_out

    # ── inference forward (same signature as WanTransformer3DModel.forward) ─
    def forward(
        self,
        input_dict,
        update_cache: int  = 0,
        cache_name:   str  = 'pos',
        action_mode:  bool = False,
        train_mode:   bool = False,
    ):
        if train_mode:
            return self.forward_train(input_dict)

        if action_mode:
            # ── action denoising ──────────────────────────────────────────
            hidden = rearrange(input_dict['noisy_latents'], 'b c f h w -> b (f h w) c')
            hidden = self.action_embedder(hidden)    # (B, L_a, d_a)

            text_hs = self.condition_embedder.text_embedder(input_dict['text_emb'])

            ts = torch.repeat_interleave(
                input_dict['timesteps'],
                input_dict['noisy_latents'].shape[-2] * input_dict['noisy_latents'].shape[-1],
                dim=1)
            temb_a, tproj = self.condition_embedder_action(ts, dtype=hidden.dtype)
            tproj = tproj.unflatten(2, (6, -1))   # (B, L_a, 6, d_a)

            # RoPE for action tokens (1D temporal; grid_id encodes t with h=w=0)
            grid_id = input_dict.get('grid_id')
            if grid_id is None:
                warnings.warn(
                    "WanMoTTransformer3DModel.forward(action_mode=True): 'grid_id' not found "
                    "in input_dict — RoPE is disabled for the action stream. "
                    "Pass input_dict['grid_id'] = get_mesh_id(..., action=True) to enable RoPE.",
                    stacklevel=2,
                )
            rotary_emb_a = self.rope(grid_id)[:, :, None] if grid_id is not None else None

            for block in self.mot_blocks:
                hidden = block.forward_action(
                    hidden, text_hs, tproj,
                    rotary_emb=rotary_emb_a,
                    update_cache=update_cache, cache_name=cache_name)

            a_tbl = self.action_scale_shift_table[None] + temb_a[:, :, None, :]
            shift, scale = rearrange(a_tbl, 'b l n c -> b n l c').chunk(2, dim=1)
            hidden = (self.action_norm_out(hidden.float()) *
                      (1. + scale.squeeze(1)) + shift.squeeze(1)).type_as(hidden)
            return self.action_proj_out(hidden)

        else:
            # ── video denoising ───────────────────────────────────────────
            hidden = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
            hidden = self.patch_embedding_mlp(hidden)   # (B, L_v, d_v)

            text_hs = self.condition_embedder.text_embedder(input_dict['text_emb'])

            grid_id = input_dict['grid_id']
            rotary_emb = self.rope(grid_id)[:, :, None]

            H, W = input_dict['noisy_latents'].shape[-2], input_dict['noisy_latents'].shape[-1]
            ts = torch.repeat_interleave(
                input_dict['timesteps'],
                (H // self.patch_size[1]) * (W // self.patch_size[2]), dim=1)
            temb, tproj = self.condition_embedder(ts, dtype=hidden.dtype)
            tproj = tproj.unflatten(2, (6, -1))   # (B, L_v, 6, d_v)

            for block in self.mot_blocks:
                hidden = block.forward_video(
                    hidden, text_hs, tproj, rotary_emb,
                    update_cache=update_cache, cache_name=cache_name)

            v_tbl = self.scale_shift_table[None] + temb[:, :, None, :]
            shift, scale = rearrange(v_tbl, 'b l n c -> b n l c').chunk(2, dim=1)
            hidden = (self.norm_out(hidden.float()) *
                      (1. + scale.squeeze(1)) + shift.squeeze(1)).type_as(hidden)
            hidden = self.proj_out(hidden)
            return rearrange(hidden, 'b l (n c) -> b (l n) c', n=math.prod(self.patch_size))


# ─────────────────────────────────────────────────────────────────────────────
def init_mot_from_video_weights(
    mot_model: WanMoTTransformer3DModel,
    video_model,   # WanTransformer3DModel
) -> None:
    """
    Initialize MoT action stream from video backbone weights (paper Section 3.3).

    Strategy per paper:
      For each action-stream weight matrix:
        1. Interpolate the corresponding video weight to the action dimension
        2. Scale by α = √(d_v / d_a)   to preserve output variance

    Video stream weights are copied 1-to-1 (same architecture, same dim).
    """
    alpha = _ALPHA
    d_v = _D_V
    d_a = mot_model._action_inner_dim

    # copy video stream weights (patch_embedding, condition_embedder, norm, proj_out)
    video_sd = video_model.state_dict()
    mot_sd   = mot_model.state_dict()

    shared_prefixes = [
        'patch_embedding_mlp', 'condition_embedder.', 'rope.',
        'norm_out', 'proj_out', 'scale_shift_table',
    ]
    for k, v in video_sd.items():
        if any(k.startswith(p) for p in shared_prefixes):
            if k in mot_sd and mot_sd[k].shape == v.shape:
                mot_sd[k] = v.clone()

    # copy/interpolate video transformer block → video stream of mot_blocks
    for layer_idx in range(len(mot_model.mot_blocks)):
        src_prefix = f'blocks.{layer_idx}.'
        dst_prefix = f'mot_blocks.{layer_idx}.'

        src_layer = {k[len(src_prefix):]: v for k, v in video_sd.items()
                     if k.startswith(src_prefix)}

        # video QKV and out: copy directly to video_to_q/k/v/to_out
        def copy_if_match(src_key, dst_key):
            sk = f'{dst_prefix}{dst_key}'
            sv = src_layer.get(src_key)
            if sv is not None and sk in mot_sd and mot_sd[sk].shape == sv.shape:
                mot_sd[sk] = sv.clone()

        # attn1 (self-attn) → video_to_q/k/v/to_out
        copy_if_match('attn1.to_q.weight', 'video_to_q.weight')
        copy_if_match('attn1.to_q.bias',   'video_to_q.bias')
        copy_if_match('attn1.to_k.weight', 'video_to_k.weight')
        copy_if_match('attn1.to_k.bias',   'video_to_k.bias')
        copy_if_match('attn1.to_v.weight', 'video_to_v.weight')
        copy_if_match('attn1.to_v.bias',   'video_to_v.bias')
        copy_if_match('attn1.to_out.0.weight', 'video_to_out.weight')
        copy_if_match('attn1.to_out.0.bias',   'video_to_out.bias')
        copy_if_match('attn1.norm_q.weight', 'video_norm_q.weight')
        copy_if_match('attn1.norm_k.weight', 'video_norm_k.weight')

        # attn2 (cross-attn) → video_cross_attn
        for sub in ['to_q', 'to_k', 'to_v', 'norm_q', 'norm_k']:
            for suf in ['weight', 'bias']:
                k_src = f'attn2.{sub}.{suf}'
                k_dst = f'video_cross_attn.{sub}.{suf}'
                sk = f'{dst_prefix}{k_dst}'
                sv = src_layer.get(k_src)
                if sv is not None and sk in mot_sd and mot_sd[sk].shape == sv.shape:
                    mot_sd[sk] = sv.clone()
        for i in range(2):
            for suf in ['weight', 'bias']:
                k_src = f'attn2.to_out.{i}.{suf}'
                k_dst = f'video_cross_attn.to_out.{i}.{suf}'
                sk = f'{dst_prefix}{k_dst}'
                sv = src_layer.get(k_src)
                if sv is not None and sk in mot_sd and mot_sd[sk].shape == sv.shape:
                    mot_sd[sk] = sv.clone()

        # video norms and FFN: copy directly
        for nm in ['video_norm1', 'video_norm2', 'video_norm3']:
            for suf in ['weight', 'bias']:
                nm_src = nm.replace('video_', '') + f'.{suf}'
                copy_if_match(nm_src, f'{nm}.{suf}')
        for suf in ['weight', 'bias']:
            copy_if_match(f'ffn.net.0.proj.{suf}', f'video_ffn.net.0.proj.{suf}')
            copy_if_match(f'ffn.net.2.{suf}',      f'video_ffn.net.2.{suf}')
        copy_if_match('scale_shift_table', 'video_scale_shift_table')

        # ── initialise action stream by interpolation from video weights ──
        def interp_to_da(w: torch.Tensor, out_dim=d_a, in_dim=None) -> torch.Tensor:
            """Bilinear interpolation of weight tensor from d_v to d_a."""
            if w.dim() == 1:
                return F.interpolate(w[None, None, :].float(), size=out_dim,
                                     mode='linear', align_corners=False)[0, 0].to(w.dtype)
            elif w.dim() == 2:
                orig_out, orig_in = w.shape
                new_out = out_dim if orig_out == d_v else orig_out
                new_in  = (in_dim or out_dim) if orig_in == d_v else orig_in
                return F.interpolate(w[None, None].float(), size=(new_out, new_in),
                                     mode='bilinear', align_corners=False)[0, 0].to(w.dtype)
            return w

        def init_action_weight(src_key, dst_key):
            sv = src_layer.get(src_key)
            dk = f'{dst_prefix}{dst_key}'
            if sv is None or dk not in mot_sd:
                return
            dv = mot_sd[dk]
            interped = interp_to_da(sv)
            if interped.shape == dv.shape:
                mot_sd[dk] = (interped * alpha).to(dv.dtype)

        # action QKV in native d_a space (§3.3): interpolate video attn1 weights
        # to [d_a, d_a] / [d_a] and scale by α (init_action_weight handles bilinear/linear interp)
        for vsrc, adst in [
            ('attn1.to_q.weight',   'action_to_q.weight'),
            ('attn1.to_q.bias',     'action_to_q.bias'),
            ('attn1.to_k.weight',   'action_to_k.weight'),
            ('attn1.to_k.bias',     'action_to_k.bias'),
            ('attn1.to_v.weight',   'action_to_v.weight'),
            ('attn1.to_v.bias',     'action_to_v.bias'),
            ('attn1.norm_q.weight', 'action_norm_q.weight'),
            ('attn1.norm_k.weight', 'action_norm_k.weight'),
        ]:
            init_action_weight(vsrc, adst)

        # per-QKV up-projections (d_a→d_v) and action_down (d_v→d_a): random init
        for key_sfx in [
            'action_q_up.weight', 'action_q_up.bias',
            'action_k_up.weight', 'action_k_up.bias',
            'action_v_up.weight', 'action_v_up.bias',
            'action_down.weight', 'action_down.bias',
        ]:
            dk = f'{dst_prefix}{key_sfx}'
            if dk in mot_sd:
                nn.init.normal_(mot_sd[dk], std=0.02)

        # action FFN interpolated from video FFN
        for suf in ['weight', 'bias']:
            init_action_weight(f'ffn.net.0.proj.{suf}', f'action_ffn.net.0.proj.{suf}')
            init_action_weight(f'ffn.net.2.{suf}',      f'action_ffn.net.2.{suf}')

        # action scale_shift_table interpolated from video
        init_action_weight('scale_shift_table', 'action_scale_shift_table')

    mot_model.load_state_dict(mot_sd, strict=False)
    print(f'[MoT init] copied video stream weights; '
          f'action stream initialised from video × α={alpha:.2f}')
