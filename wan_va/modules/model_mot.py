# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Mixture-of-Transformers (MoT) model for LingBot-VA — dv-resident action stream,
UNIFIED video+action KV cache (paper-correct interleaved sequence).

Design (confirmed):
  - Video stream:  d_v=3072, native dimension, unchanged from shared backbone.
  - Action stream: enters the network at d_v=3072 (no d_a intermediate bottleneck).
        raw action (30) --self.action_embedder--> d_v=3072
        ... 30 MoT blocks, action stays in d_v the entire time ...
        d_v=3072 --self.action_proj_out--> raw action (30)
  - Inside every block, action has its OWN independent attn1 / attn2 / ffn / norms,
    all at d_v=3072 (same shape as the video stream's). These are NOT shared
    with the video stream's modules — they are separate nn.Module instances.
    At conversion time (see the separately written conversion script) their
    weights are initialized as exact clones of the video stream's weights,
    then allowed to diverge freely during training.
  - FFN: action has its own d_v-sized FFN (same shape as video's FFN).

  This intentionally departs from the paper's da<<dv lightweight-action design
  in exchange for maximal, lossless reuse of pretrained shared-backbone weights:
  every action-stream weight in this file has an exact-shape counterpart in the
  shared backbone checkpoint (WanTransformer3DModel), so no interpolation or
  alpha-scaling is required at conversion time — only weight cloning.

UNIFIED KV CACHE (critical, paper Section 3.2/3.3):
  Paper: "forming a unified sequence [zt, at,1, ..., at,tau, zt+1, ...] for
  joint modeling" and "our method unifies video and action within a single
  causal autoregressive framework, enabling persistent memory through KV
  cache". Video and action tokens are NOT two independently-cached streams —
  they share ONE interleaved sequence, so:
    - action tokens can attend the full history of BOTH video and action
    - video tokens can attend the full history of BOTH video and action
    - tokens within the SAME chunk (the K video frames / tau*K action tokens
      currently being generated) attend each other bidirectionally
    - tokens from PAST chunks are attended causally
  Each WanMoTTransformerBlock owns exactly one shared cache pool (one
  WanAttention.attn_caches dict, created on attn1's __init__ and then
  aliased onto action_attn1.attn_caches — see the assignment at the end of
  WanMoTTransformerBlock.__init__). Both attn1 (video QKV weights) and
  action_attn1 (action QKV weights) read/write into that same pool, so
  self-attention naturally sees the full unified history.

  Simplification accepted for this implementation (vs. the shared backbone's
  full FlexAttn frame_id/noise_id mask machinery): we reuse the existing
  WanAttention.update_cache / allocate_slots / restore_cache mechanism as-is.
  All tokens belonging to the SAME chunk currently being generated are passed
  into a single forward call together (matching how the video chunk and the
  tau-action chunk are each generated as one batched flow-matching call), so
  they attend each other natively within that one attention op — no separate
  bidirectional mask is needed for the "current chunk" case. Tokens from
  earlier chunks are already resident in the cache and are attended causally
  (every cached token was written by an earlier, completed forward call).

Training:  joint cross-modal self-attention (separate action_* modules, video
           modules untouched), reusing the same FlexAttn mask machinery as the
           shared backbone (mask is purely positional/token-count based).
Inference: video chunk is generated first (forward_video), then the action
           chunk is generated conditioned on it (forward_action) — matching
           paper Algorithm 1 Lines 5/7. Both write into the SAME shared cache
           pool per block, so action generation can attend the video chunk
           that was just written, plus all prior video+action history.
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
    custom_sdpa,
)

__all__ = ['WanMoTTransformer3DModel']


class WanMoTTransformerBlock(nn.Module):
    """
    One MoT layer. Video stream is architecturally identical to the shared
    backbone's WanTransformerBlock. Action stream is a parallel, independently
    parameterized copy of the same block shape (all at d_v), fused with video
    through a SHARED, UNIFIED KV cache pool (see module docstring).
    """

    def __init__(
        self,
        dim,            # d_v = 3072
        ffn_dim,        # 14336
        num_heads,      # 24
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode
        self.dim = dim

        # ── Video stream (identical to shared backbone) ────────────────────
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads,
            eps=eps, cross_attention_dim_head=None, attn_mode=attn_mode,
        )
        self.attn2 = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads,
            eps=eps, cross_attention_dim_head=dim // num_heads, attn_mode=attn_mode,
        )
        self.norm2 = (FP32LayerNorm(dim, eps, elementwise_affine=True)
                     if cross_attn_norm else nn.Identity())
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # ── Action stream — same shapes as video, independent parameters ───
        self.action_norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.action_attn1 = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads,
            eps=eps, cross_attention_dim_head=None, attn_mode=attn_mode,
        )
        # Cross-attention with text — text is already in d_v, no projection needed
        self.action_attn2 = WanAttention(
            dim=dim, heads=num_heads, dim_head=dim // num_heads,
            eps=eps, cross_attention_dim_head=dim // num_heads, attn_mode=attn_mode,
        )
        self.action_norm2 = (FP32LayerNorm(dim, eps, elementwise_affine=True)
                             if cross_attn_norm else nn.Identity())
        self.action_ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.action_norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.action_scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # ── UNIFIED self-attention cache pool ───────────────────────────────
        # attn1.attn_caches is the dict that actually holds the cache pool
        # (created in WanAttention.__init__ since cross_attention_dim_head=None
        # for both attn1 and action_attn1). We alias action_attn1.attn_caches
        # to point at the SAME dict object, so update_cache()/allocate_slots()
        # calls from either attn1 or action_attn1 read/write the same pool —
        # giving video and action tokens a truly unified, interleaved history.
        self.action_attn1.attn_caches = self.attn1.attn_caches

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _unpack_adaln(scale_shift_table, temb):
        """scale_shift_table: [1,6,C], temb: [B,L,6,C] -> 6 x [B,L,C]."""
        tbl = scale_shift_table[None] + temb.float()
        parts = rearrange(tbl, 'b l n c -> b n l c').chunk(6, dim=1)
        return [p.squeeze(1) for p in parts]

    # FSDP2 pre-forward (unshard) and activation-checkpointing hooks only fire
    # on __call__ — forward_train must invoke the block, not forward_joint
    # directly, or sharded DTensor params leak into plain-tensor math.
    def forward(self, *args, **kwargs):
        return self.forward_joint(*args, **kwargs)

    @staticmethod
    def _cross_attn_sdpa(attn, x, context):
        """Text cross-attention over exact-length (unpadded) text tokens.

        FlexAttnFunc.cross_attention_mask is built for the shared-backbone
        layout (full concatenated video+action sequence x 512 padded text
        tokens); forward_joint cross-attends the two streams separately with
        unpadded text, where masking (batch=1, no text padding) degenerates
        to full attention — plain SDPA is exact.
        """
        q = attn.norm_q(attn.to_q(x)).unflatten(2, (attn.heads, -1))
        k = attn.norm_k(attn.to_k(context)).unflatten(2, (attn.heads, -1))
        v = attn.to_v(context).unflatten(2, (attn.heads, -1))
        out = custom_sdpa(q, k, v).flatten(2, 3).type_as(x)
        return attn.to_out[1](attn.to_out[0](out))

    # ── training: joint cross-modal forward (video || action concatenated) ─
    def forward_joint(
        self,
        video_states: torch.Tensor,           # (1, L_v, d_v)
        action_states: torch.Tensor,          # (1, L_a, d_v)  — already in d_v
        encoder_hidden_states: torch.Tensor,  # (1, L_t, d_v)
        video_temb: torch.Tensor,             # (1, L_v, 6, d_v)
        action_temb: torch.Tensor,            # (1, L_a, 6, d_v)
        rotary_emb: torch.Tensor,             # (1, L_v+L_a, 1, head_dim)
    ):
        """
        Used by forward_train(). No KV cache is used here — the whole episode
        (history + current chunk, video and action interleaved) is processed
        in one padded forward pass per the shared backbone's training scheme,
        with causal/bidirectional structure enforced by FlexAttnFunc's block
        mask (see WanMoTTransformer3DModel.forward_train), not by this method.
        Video and action tokens are concatenated into one sequence and run
        through self-attention together: video contributes Q/K/V via attn1's
        weights, action contributes Q/K/V via action_attn1's weights (separate
        parameters, same d_v shape), and the concatenated set attends jointly.
        """
        L_v = video_states.shape[1]
        L_a = action_states.shape[1]

        # ── adaLN ────────────────────────────────────────────────────────
        sh_v, sc_v, g_v, c_sh_v, c_sc_v, c_g_v = self._unpack_adaln(
            self.scale_shift_table, video_temb)
        sh_a, sc_a, g_a, c_sh_a, c_sc_a, c_g_a = self._unpack_adaln(
            self.action_scale_shift_table, action_temb)

        norm_v = (self.norm1(video_states.float()) * (1. + sc_v) + sh_v).type_as(video_states)
        norm_a = (self.action_norm1(action_states.float()) * (1. + sc_a) + sh_a).type_as(action_states)

        # ── self-attention: compute video Q/K/V via attn1, action Q/K/V via
        #    action_attn1 (independent weights, same d_v shape), concatenate
        #    the projected Q/K/V tensors, run attention jointly. The resulting
        #    QK pair spans the FULL unified video+action sequence, matching
        #    the paper's interleaved-sequence formulation.
        #
        #    CAUSAL MASKING (critical — see module docstring "Training" note):
        #    self.attn1.attn_op is resolved at __init__ time from attn_mode.
        #    When attn_mode='flex', attn_op is a FlexAttnFunc instance whose
        #    .forward() reads the CLASS-LEVEL attribute FlexAttnFunc.attention_mask
        #    (set by FlexAttnFunc.init_mask(), called in forward_train below)
        #    and applies it internally — no mask argument needs to be passed
        #    here, exactly mirroring how the shared backbone's WanAttention.
        #    forward() relies on the same mechanism. If attn_mode is 'torch'
        #    or 'flashattn' instead, attn_op is custom_sdpa / flash_attn_func,
        #    NEITHER of which accepts or applies any mask — attention would
        #    silently become fully bidirectional with NO causal structure,
        #    violating the paper's core autoregressive formulation (Fig. 3).
        #    We assert this explicitly rather than fail silently. ─────────────
        assert self.attn_mode == 'flex', (
            "WanMoTTransformerBlock.forward_joint() requires attn_mode='flex' "
            "for the causal/bidirectional training mask (FlexAttnFunc.attention_mask, "
            "set by FlexAttnFunc.init_mask() in forward_train) to actually be "
            f"applied. Got attn_mode='{self.attn_mode}', whose attn_op "
            "(custom_sdpa/flash_attn_func) ignores masking entirely — training "
            "would silently run with full bidirectional attention, contradicting "
            "the paper's causal autoregressive design (Section 3.2, Figure 3)."
        )
        q_v = self.attn1.to_q(norm_v)
        k_v = self.attn1.to_k(norm_v)
        v_v = self.attn1.to_v(norm_v)
        q_v = self.attn1.norm_q(q_v).unflatten(2, (self.attn1.heads, -1))
        k_v = self.attn1.norm_k(k_v).unflatten(2, (self.attn1.heads, -1))
        v_v = v_v.unflatten(2, (self.attn1.heads, -1))

        q_a = self.action_attn1.to_q(norm_a)
        k_a = self.action_attn1.to_k(norm_a)
        v_a = self.action_attn1.to_v(norm_a)
        q_a = self.action_attn1.norm_q(q_a).unflatten(2, (self.action_attn1.heads, -1))
        k_a = self.action_attn1.norm_k(k_a).unflatten(2, (self.action_attn1.heads, -1))
        v_a = v_a.unflatten(2, (self.action_attn1.heads, -1))

        def _apply_rope(x, freqs):
            x_c = torch.view_as_complex(
                x.to(torch.float64).reshape(*x.shape[:3], -1, 2))
            return torch.view_as_real(x_c * freqs).flatten(3).to(x.dtype)

        if rotary_emb is not None:
            q_v = _apply_rope(q_v, rotary_emb[:, :L_v])
            k_v = _apply_rope(k_v, rotary_emb[:, :L_v])
            q_a = _apply_rope(q_a, rotary_emb[:, L_v:L_v + L_a])
            k_a = _apply_rope(k_a, rotary_emb[:, L_v:L_v + L_a])

        # NOTE: order here (video first, action second) must match the order
        # FlexAttnFunc.init_mask used to build seq_ids/frame_ids/noise_ids in
        # forward_train, since the block mask indexes by flat position.
        q = torch.cat([q_v, q_a], dim=1)
        k = torch.cat([k_v, k_a], dim=1)
        v = torch.cat([v_v, v_a], dim=1)

        # attn1.attn_op here IS a FlexAttnFunc(is_cross=False) instance (see
        # assert above), so this call internally applies FlexAttnFunc.attention_mask
        # — the causal/chunk-bidirectional mask built by init_mask() — exactly
        # as the shared backbone's WanAttention.forward does for self-attention.
        attn_out = self.attn1.attn_op(q, k, v)
        attn_out = attn_out.flatten(2, 3).type_as(video_states)

        out_v = self.attn1.to_out[1](self.attn1.to_out[0](attn_out[:, :L_v]))
        out_a = self.action_attn1.to_out[1](self.action_attn1.to_out[0](attn_out[:, L_v:]))

        video_states = (video_states.float() + out_v.float() * g_v).type_as(video_states)
        action_states = (action_states.float() + out_a.float() * g_a).type_as(action_states)

        # ── cross-attention with text — both streams in d_v, no projection ──
        # self.attn2 / self.action_attn2 here go through WanAttention.forward()
        # UNMODIFIED (unlike self-attention above), so when attn_mode='flex'
        # their attn_op is FlexAttnFunc(is_cross=True), which reads
        # FlexAttnFunc.cross_attention_mask internally — same mechanism as
        # the shared backbone's cross-attention, no extra wiring needed here.
        norm_v2 = self.norm2(video_states.float()).type_as(video_states)
        video_states = video_states + self._cross_attn_sdpa(
            self.attn2, norm_v2, encoder_hidden_states)

        norm_a2 = self.action_norm2(action_states.float()).type_as(action_states)
        action_states = action_states + self._cross_attn_sdpa(
            self.action_attn2, norm_a2, encoder_hidden_states)

        # ── FFN — separate per stream, both d_v-sized ───────────────────────
        norm_v3 = (self.norm3(video_states.float()) * (1. + c_sc_v) + c_sh_v).type_as(video_states)
        video_states = (video_states.float() + self.ffn(norm_v3).float() * c_g_v).type_as(video_states)

        norm_a3 = (self.action_norm3(action_states.float()) * (1. + c_sc_a) + c_sh_a).type_as(action_states)
        action_states = (action_states.float() + self.action_ffn(norm_a3).float() * c_g_a).type_as(action_states)

        return video_states, action_states

    # ── inference: video chunk generation (Algorithm 1 Line 5) ─────────────
    # Writes video K/V into the UNIFIED cache pool (self.attn1.attn_caches,
    # aliased to self.action_attn1.attn_caches). A later forward_action call
    # for the SAME chunk will therefore see this video chunk's K/V already
    # resident in the pool, satisfying the inverse-dynamics conditioning
    # on z_t / ẑ_{t+1} (paper Eq. 9) and the unified-sequence design.
    def forward_video(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        update_cache: int = 0,
        cache_name: str = 'pos',
    ) -> torch.Tensor:
        sh, sc, g, c_sh, c_sc, c_g = self._unpack_adaln(self.scale_shift_table, temb)

        norm_h = (self.norm1(hidden_states.float()) * (1. + sc) + sh).type_as(hidden_states)
        attn_out = self.attn1(norm_h, norm_h, norm_h, rotary_emb,
                              update_cache=update_cache, cache_name=cache_name)
        hidden_states = (hidden_states.float() + attn_out * g).type_as(hidden_states)

        norm_h = self.norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + self.attn2(
            norm_h, encoder_hidden_states, encoder_hidden_states, None,
            update_cache=0, cache_name=cache_name)

        norm_h = (self.norm3(hidden_states.float()) * (1. + c_sc) + c_sh).type_as(hidden_states)
        hidden_states = (hidden_states.float() + self.ffn(norm_h).float() * c_g).type_as(hidden_states)
        return hidden_states

    # ── inference: action chunk generation (Algorithm 1 Line 7) ────────────
    # IMPORTANT: cache_name here MUST match the cache_name used in the
    # forward_video call for this same chunk (default 'pos' for both), since
    # the pool is shared by dict identity but indexed by this string key.
    # Action self-attention therefore attends:
    #   - all prior chunks' video AND action K/V (written by earlier
    #     forward_video/forward_action calls with update_cache!=0)
    #   - the CURRENT chunk's video K/V (just written by forward_video,
    #     above, in this same autoregressive step)
    #   - the CURRENT chunk's own action tokens (all tau*K of them, since
    #     they are passed into this one forward call together and attend
    #     each other natively within the attention op — see module docstring)
    def forward_action(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        update_cache: int = 0,
        cache_name: str = 'pos',
    ) -> torch.Tensor:
        sh, sc, g, c_sh, c_sc, c_g = self._unpack_adaln(self.action_scale_shift_table, temb)

        norm_h = (self.action_norm1(hidden_states.float()) * (1. + sc) + sh).type_as(hidden_states)
        attn_out = self.action_attn1(norm_h, norm_h, norm_h, rotary_emb,
                                     update_cache=update_cache, cache_name=cache_name)
        hidden_states = (hidden_states.float() + attn_out * g).type_as(hidden_states)

        norm_h = self.action_norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + self.action_attn2(
            norm_h, encoder_hidden_states, encoder_hidden_states, None,
            update_cache=0, cache_name=cache_name)

        norm_h = (self.action_norm3(hidden_states.float()) * (1. + c_sc) + c_sh).type_as(hidden_states)
        hidden_states = (hidden_states.float() + self.action_ffn(norm_h).float() * c_g).type_as(hidden_states)
        return hidden_states


class WanMoTTransformer3DModel(ModelMixin, ConfigMixin):
    """
    MoT variant of the shared backbone. Drop-in replacement: forward() /
    forward_train() signatures match WanTransformer3DModel, so callers
    (e.g. VA_Server) require no changes beyond pointing at this class.

    Action stream design (confirmed):
      - No d_a bottleneck. self.action_embedder / self.action_proj_out map
        directly between raw action_dim (30) and d_v (3072).
      - Every block has independent action_* weights at d_v, structurally
        identical in shape to the video weights — enabling lossless weight
        cloning from a shared-backbone checkpoint (see the separately
        written conversion script).
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "norm",
    ]
    _no_split_modules = ["WanMoTTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "action_scale_shift_table_final",
        "norm1", "action_norm1", "text_norm1",
        "norm2", "action_norm2", "text_norm2",
        "norm3", "action_norm3", "text_norm3",
        "norm_out", "action_norm_out",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanMoTTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 action_dim=30,
                 text_dim=4096,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch"):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim   # d_v = 3072

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)

        # ── video embedding (identical to shared backbone) ─────────────────
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim, time_freq_dim=freq_dim, time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim, pos_embed_seq_len=pos_embed_seq_len,
        )

        # ── action embedding — same shapes as shared backbone's
        #    action_embedder / action_proj_out (30 <-> d_v, no d_a step).
        #    This means these two layers can be cloned 1:1 from a shared
        #    backbone checkpoint with zero modification. ────────────────────
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        # ── MoT blocks ───────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            WanMoTTransformerBlock(
                inner_dim, ffn_dim, num_attention_heads,
                cross_attn_norm, eps, attn_mode=attn_mode,
            ) for _ in range(num_layers)
        ])

        # ── video output head (identical to shared backbone) ───────────────
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # ── action output head — same shape as shared backbone's
        #    final norm/scale_shift_table (d_v), so it too clones 1:1. ──────
        self.action_norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.action_scale_shift_table_final = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        # self.action_proj_out (defined above) reused as the final action head.

    # ── cache management ─────────────────────────────────────────────────
    # NOTE: attn1.attn_caches and action_attn1.attn_caches are the SAME dict
    # object (aliased in WanMoTTransformerBlock.__init__), so each of these
    # only needs to be called once per block via attn1 — calling it again via
    # action_attn1 would re-initialize and wipe the pool attn1 just set up.
    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)

    # ── embedding helpers ────────────────────────────────────────────────
    def _video_embed(self, latents):
        x = rearrange(
            latents, 'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
            p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
        return self.patch_embedding_mlp(x)

    def _action_embed(self, actions):
        x = rearrange(actions, 'b c f h w -> b (f h w) c')
        return self.action_embedder(x)   # (B, L_a, d_v) — directly, no d_a step

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        ts = torch.repeat_interleave(
            timesteps, (H // pach_scale_h) * (W // pach_scale_w), dim=1)
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder)
        temb, timestep_proj = current_condition_embedder(ts, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return temb, timestep_proj

    # ── training forward ─────────────────────────────────────────────────
    def forward_train(self, input_dict):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent']        = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent']        = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size  = latent_dict['noisy_latents'].shape[0]
        # _cross_attn_sdpa runs UNMASKED over exact-length text — only valid
        # when the flat sequence holds a single sample (no cross-sample leak).
        assert batch_size == 1, (
            'forward_train cross-attention assumes batch_size=1 per rank; '
            'multi-sample packing needs a per-sample cross mask')

        # ── embed ────────────────────────────────────────────────────────
        v_noise = self._video_embed(latent_dict['noisy_latents']).flatten(0, 1)[None]
        v_cond  = self._video_embed(latent_dict['latent']).flatten(0, 1)[None]
        a_noise = self._action_embed(action_dict['noisy_latents']).flatten(0, 1)[None]
        a_cond  = self._action_embed(action_dict['latent']).flatten(0, 1)[None]
        text_hs = self.condition_embedder.text_embedder(
            latent_dict["text_emb"]).flatten(0, 1)[None]

        video_states  = torch.cat([v_noise, v_cond], dim=1)   # (1, L_v, d_v)
        action_states = torch.cat([a_noise, a_cond], dim=1)   # (1, L_a, d_v)

        # ── rotary (video and action share the same head_dim, so reuse rope) ─
        lat_grid = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        act_grid = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid = torch.cat([lat_grid] * 2 + [act_grid] * 2, dim=2)
        rotary_emb = self.rope(full_grid)[:, :, None]

        # ── time embeddings ─────────────────────────────────────────────
        lat_ts = torch.cat([latent_dict['timesteps'].flatten(0, 1),
                            latent_dict['cond_timesteps'].flatten(0, 1)])[None]
        act_ts = torch.cat([action_dict['timesteps'].flatten(0, 1),
                            action_dict['cond_timesteps'].flatten(0, 1)])[None]

        H_lat, W_lat = latent_dict['noisy_latents'].shape[-2:]
        H_act, W_act = action_dict['noisy_latents'].shape[-2:]

        latent_temb, latent_tproj = self._time_embed(
            lat_ts, H_lat, W_lat, dtype=video_states.dtype, action_mode=False)
        action_temb, action_tproj = self._time_embed(
            act_ts, H_act, W_act, dtype=action_states.dtype, action_mode=True)

        # ── pad to a multiple of 128 (required for FlexAttn's compiled
        #    block_mask to align with the actual sequence length — same
        #    requirement as the shared backbone's forward_train). ──────────
        L_v_total = video_states.shape[1]
        L_a_total = action_states.shape[1]
        total_length = L_v_total + L_a_total
        padded_length = (128 - total_length % 128) % 128

        if padded_length > 0:
            # Pad onto the action stream's tail (matches shared backbone's
            # convention of appending pad tokens after the real sequence).
            action_states = F.pad(action_states, (0, 0, 0, padded_length))
            rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
            action_tproj = F.pad(action_tproj, (0, 0, 0, 0, 0, padded_length))

        # ── FlexAttn mask (same machinery as shared backbone) ──────────────
        FlexAttnFunc.init_mask(
            latent_dict['noisy_latents'].shape, action_dict['noisy_latents'].shape,
            padded_length=padded_length,
            chunk_size=input_dict["chunk_size"],
            window_size=input_dict['window_size'],
            patch_size=self.patch_size,
            device=video_states.device,
        )

        # ── block loop ──────────────────────────────────────────────────
        for block in self.blocks:
            # Module call (NOT block.forward_joint) so FSDP unshard and
            # activation-checkpointing wrappers actually engage.
            video_states, action_states = block(
                video_states, action_states, text_hs,
                latent_tproj, action_tproj, rotary_emb,
            )

        # ── video output head ───────────────────────────────────────────
        L_v_noisy = v_noise.shape[1]
        v_sst = self.scale_shift_table[None] + latent_temb[:, :, None, ...]
        shift_v, scale_v = [x.squeeze(1) for x in
                            rearrange(v_sst, 'b l n c -> b n l c').chunk(2, dim=1)]
        v_out = (self.norm_out(video_states.float()) * (1. + scale_v) + shift_v).type_as(video_states)
        v_out = self.proj_out(v_out[:, :L_v_noisy])
        v_out = rearrange(v_out, '1 (b l) (n c) -> b (l n) c',
                          n=math.prod(self.patch_size), b=batch_size)

        # ── action output head ──────────────────────────────────────────
        # action_states is padded (length = L_a_total + padded_length) but
        # action_temb is NOT (it was computed before padding, at the
        # original a_noise+a_cond length). Slice action_states down to the
        # unpadded length BEFORE combining with action_temb to avoid a
        # shape mismatch in the broadcast add below.
        L_a_noisy  = a_noise.shape[1]
        L_a_total  = a_noise.shape[1] + a_cond.shape[1]
        action_states_unpadded = action_states[:, :L_a_total]

        a_sst = self.action_scale_shift_table_final[None] + action_temb[:, :, None, ...]
        shift_a, scale_a = [x.squeeze(1) for x in
                            rearrange(a_sst, 'b l n c -> b n l c').chunk(2, dim=1)]
        a_out = (self.action_norm_out(action_states_unpadded.float()) *
                (1. + scale_a) + shift_a).type_as(action_states)
        a_out = self.action_proj_out(a_out[:, :L_a_noisy])
        a_out = rearrange(a_out, '1 (b l) c -> b l c', b=batch_size)

        return v_out, a_out

    # ── inference forward (signature matches shared backbone) ──────────────
    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        if train_mode:
            return self.forward_train(input_dict)

        if action_mode:
            hidden = rearrange(input_dict['noisy_latents'], 'b c f h w -> b (f h w) c')
            hidden = self.action_embedder(hidden)   # raw(30) -> d_v directly
        else:
            hidden = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0], p2=self.patch_size[1], p3=self.patch_size[2])
            hidden = self.patch_embedding_mlp(hidden)

        text_hidden_states = self.condition_embedder.text_embedder(input_dict["text_emb"])

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder)
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=hidden.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))

        for block in self.blocks:
            if action_mode:
                hidden = block.forward_action(
                    hidden, text_hidden_states, timestep_proj, rotary_emb,
                    update_cache=update_cache, cache_name=cache_name)
            else:
                hidden = block.forward_video(
                    hidden, text_hidden_states, timestep_proj, rotary_emb,
                    update_cache=update_cache, cache_name=cache_name)

        if action_mode:
            a_sst = self.action_scale_shift_table_final[None] + temb[:, :, None, ...]
            shift, scale = [x.squeeze(1) for x in
                            rearrange(a_sst, 'b l n c -> b n l c').chunk(2, dim=1)]
            hidden = (self.action_norm_out(hidden.float()) * (1. + scale) + shift).type_as(hidden)
            hidden = self.action_proj_out(hidden)   # d_v -> raw(30) directly
        else:
            v_sst = self.scale_shift_table[None] + temb[:, :, None, ...]
            shift, scale = [x.squeeze(1) for x in
                            rearrange(v_sst, 'b l n c -> b n l c').chunk(2, dim=1)]
            hidden = (self.norm_out(hidden.float()) * (1. + scale) + shift).type_as(hidden)
            hidden = self.proj_out(hidden)
            hidden = rearrange(hidden, 'b l (n c) -> b (l n) c', n=math.prod(self.patch_size))

        return hidden


if __name__ == '__main__':
    model = WanMoTTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\nTotal parameters: {n_params / 1e9:.2f}B")