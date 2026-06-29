# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

def flash_attn_func(*args, **kwargs):
    # Lazy import: only loaded when attn_mode='flashattn'
    try:
        from flash_attn_interface import flash_attn_func as _fa
    except ImportError:
        from flash_attn import flash_attn_func as _fa
    return _fa(*args, **kwargs)

__all__ = ['WanTransformer3DModel']


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True, 
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self, 
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = FlexAttnFunc.cross_attention_mask if self.is_cross else FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape, 
        action_shape, 
        padded_length, 
        chunk_size,
        window_size,
        patch_size,
        device,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        latent_seq_id = torch.arange(B)[:, None, None, None].\
            expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]).flatten()
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = torch.arange(L_F)[None, :, None, None].expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None].flatten()
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2)

        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, 512).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross
    
    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
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
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
        action_hidden_dim: int = 768,
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Video Expert FFN
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

        # 4. Action Expert — full da=768 stream (paper-correct architecture)
        # da/dv ratio = 768/3072 = 1/4; da_ffn = ffn_dim * (da/dv) = 3584
        da = action_hidden_dim
        dv = dim
        da_ffn = int(ffn_dim * da / dv)          # 14336 × (768/3072) = 3584
        self.action_hidden_dim = da

        # Separate QKV projections: action input at da → joint attention at dv
        self.action_to_q   = nn.Linear(da, dv)
        self.action_to_k   = nn.Linear(da, dv)
        self.action_to_v   = nn.Linear(da, dv)
        self.action_to_out = nn.Linear(dv, da)   # project attention output back to da

        # Pre-self-attention norm for action tokens at da
        self.action_norm1 = FP32LayerNorm(da, eps, elementwise_affine=False)

        # Separate Q projection for action cross-attention (keys/values shared with video)
        self.action_cross_to_q   = nn.Linear(da, dv)
        self.action_cross_to_out = nn.Linear(dv, da)
        self.action_norm2 = (FP32LayerNorm(da, eps, elementwise_affine=True)
                             if cross_attn_norm else nn.Identity())

        # Action FFN: [da → da_ffn → da] = [768 → 3584 → 768]
        self.action_ffn   = FeedForward(da, inner_dim=da_ffn, activation_fn="gelu-approximate")
        self.action_norm3 = FP32LayerNorm(da, eps, elementwise_affine=False)

        # AdaLN modulation at da (6 params: shift/scale/gate × SA + FFN)
        self.action_scale_shift_table = nn.Parameter(
            torch.randn(1, 6, da) / da**0.5)

    def forward(
        self,
        hidden_states,           # video tokens [B, L_v, dv], or action tokens [B, L_a, da] in action_only
        encoder_hidden_states,   # text [B, L_text, text_dim]
        temb,                    # AdaLN timestep proj: video [B, L_v, 6, dv] or action [B, L_a, 6, da]
        rotary_emb,              # [1, L, 1, C]
        update_cache=0,
        cache_name='pos',
        # Two-stream MoT training args:
        action_states=None,      # [B, L_a, da=768] — None ⇒ single-stream path
        temb_action=None,        # [B, L_a, 6, da=768]
        # Single-stream action-only inference:
        action_only=False,       # True ⇒ run action expert layers on hidden_states at da=768
    ):
        # ── helpers ──────────────────────────────────────────────────────────
        def _adaLN6(sst, temb_):
            """Unpack 6 AdaLN modulation params from scale_shift_table + temb."""
            out = rearrange(sst[None] + temb_.float(), 'b l n c -> b n l c').chunk(6, dim=1)
            return [x.squeeze(1) for x in out]  # each [B, L, C]

        def _apply_rope(x, freqs):
            x_c = torch.view_as_complex(
                x.to(torch.float64).reshape(*x.shape[:3], -1, 2))
            return torch.view_as_real(x_c * freqs).flatten(3).to(x.dtype)

        # ── Action-only single-stream inference path (da=768) ────────────
        if action_only:
            h_a = hidden_states   # [B, L_a, da=768]
            nh, hd = self.attn1.heads, self.attn1.inner_dim // self.attn1.heads
            sh_a, sc_a, g_a, c_sh_a, c_sc_a, c_g_a = _adaLN6(self.action_scale_shift_table, temb)

            # Original backbone weights may be DTensors (FSDP inference sharding).
            # Action expert weights are plain tensors. Use local helpers to avoid mixing.
            def _local(t):
                """Gather a DTensor to a plain local tensor; no-op for regular tensors."""
                if hasattr(t, 'full_tensor'):
                    return t.full_tensor()
                if hasattr(t, 'to_local'):
                    return t.to_local()
                return t

            def _linear_local(x, lin):
                return F.linear(x, _local(lin.weight).to(x.dtype),
                                _local(lin.bias).to(x.dtype) if lin.bias is not None else None)

            def _rms_norm_local(x, rms_mod):
                return F.rms_norm(x, rms_mod.normalized_shape,
                                  _local(rms_mod.weight).to(x.dtype), rms_mod.eps)

            # Self-attention using action QKV projections (da→dv)
            h_a_n = (self.action_norm1(h_a.float()) * (1. + sc_a) + sh_a).type_as(h_a)
            q_a = _rms_norm_local(self.action_to_q(h_a_n), self.attn1.norm_q).unflatten(2, (nh, hd))
            k_a = _rms_norm_local(self.action_to_k(h_a_n), self.attn1.norm_k).unflatten(2, (nh, hd))
            v_a = self.action_to_v(h_a_n).unflatten(2, (nh, hd))
            if rotary_emb is not None:
                q_a = _apply_rope(q_a, rotary_emb)
                k_a = _apply_rope(k_a, rotary_emb)
            attn_out_a = custom_sdpa(q_a, k_a, v_a).flatten(2)   # [B, L_a, dv]
            h_a = (h_a.float() + self.action_to_out(attn_out_a).float() * g_a).type_as(h_a)

            # Cross-attention with text (K/V via attn2 — use _linear_local for DTensor weights)
            h_a_n2 = self.action_norm2(h_a.float()).type_as(h_a)
            q_ac = _rms_norm_local(self.action_cross_to_q(h_a_n2), self.attn2.norm_q).unflatten(2, (nh, hd))
            k_txt_flat = _linear_local(encoder_hidden_states, self.attn2.to_k)
            k_txt = _rms_norm_local(k_txt_flat, self.attn2.norm_k).unflatten(2, (nh, hd))
            v_txt = _linear_local(encoder_hidden_states, self.attn2.to_v).unflatten(2, (nh, hd))
            cross_out = custom_sdpa(q_ac, k_txt, v_txt).flatten(2)
            h_a = h_a + self.action_cross_to_out(cross_out)

            # FFN
            h_a_n3 = (self.action_norm3(h_a.float()) * (1. + c_sc_a) + c_sh_a).type_as(h_a)
            h_a = (h_a.float() + self.action_ffn(h_a_n3).float() * c_g_a).type_as(h_a)

            return h_a, None

        # ── Two-stream MoT training path ──────────────────────────────────
        if action_states is not None:
            h_v = hidden_states   # [B, L_v, dv]
            h_a = action_states   # [B, L_a, da]
            L_v = h_v.shape[1]

            # Unpack video AdaLN (6 params at dv)
            sh_v, sc_v, g_v, c_sh_v, c_sc_v, c_g_v = _adaLN6(self.scale_shift_table, temb)
            # Unpack action AdaLN (6 params at da)
            sh_a, sc_a, g_a, c_sh_a, c_sc_a, c_g_a = _adaLN6(self.action_scale_shift_table, temb_action)

            # ── 1. Joint self-attention with separate QKV projections ────────
            h_v_n = (self.norm1(h_v.float()) * (1. + sc_v) + sh_v).type_as(h_v)
            h_a_n = (self.action_norm1(h_a.float()) * (1. + sc_a) + sh_a).type_as(h_a)

            # Video QKV via shared attn1 projections [dv → dv]
            q_v = self.attn1.to_q(h_v_n)
            k_v = self.attn1.to_k(h_v_n)
            v_v = self.attn1.to_v(h_v_n)

            # Action QKV via separate projections [da → dv]
            q_a = self.action_to_q(h_a_n)
            k_a = self.action_to_k(h_a_n)
            v_a = self.action_to_v(h_a_n)

            # Concatenate joint sequence at dv
            q = torch.cat([q_v, q_a], dim=1)
            k = torch.cat([k_v, k_a], dim=1)
            v = torch.cat([v_v, v_a], dim=1)

            # QK norm + reshape to heads
            nh, hd = self.attn1.heads, self.attn1.inner_dim // self.attn1.heads
            q = self.attn1.norm_q(q).unflatten(2, (nh, hd))
            k = self.attn1.norm_k(k).unflatten(2, (nh, hd))
            v = v.unflatten(2, (nh, hd))

            # RoPE (combined positions cover L_v + L_a)
            if rotary_emb is not None:
                q = _apply_rope(q, rotary_emb)
                k = _apply_rope(k, rotary_emb)

            # Attention — use standard SDPA (avoids FlexAttn mask-size constraints)
            attn_out = custom_sdpa(q, k, v).flatten(2)  # [B, L_v+L_a, dv]

            # Split + project out
            out_v = self.attn1.to_out[0](attn_out[:, :L_v])
            out_v = self.attn1.to_out[1](out_v)              # dropout (no-op in eval)
            out_a = self.action_to_out(attn_out[:, L_v:])    # [B, L_a, da]

            h_v = (h_v.float() + out_v.float() * g_v).type_as(h_v)
            h_a = (h_a.float() + out_a.float() * g_a).type_as(h_a)

            # ── 2. Cross-attention (text conditioning) ───────────────────────
            # Video — unchanged through attn2
            h_v_n2 = self.norm2(h_v.float()).type_as(h_v)
            h_v = h_v + self.attn2(h_v_n2, encoder_hidden_states, encoder_hidden_states,
                                   None, update_cache=0, cache_name=cache_name)

            # Action — separate Q projection + shared K/V from text
            h_a_n2 = self.action_norm2(h_a.float()).type_as(h_a)
            q_ac = self.action_cross_to_q(h_a_n2).unflatten(2, (nh, hd))
            q_ac = self.attn2.norm_q(q_ac.flatten(2)).unflatten(2, (nh, hd))
            k_txt = self.attn2.to_k(encoder_hidden_states)
            v_txt = self.attn2.to_v(encoder_hidden_states)
            k_txt = self.attn2.norm_k(k_txt).unflatten(2, (nh, hd))
            v_txt = v_txt.unflatten(2, (nh, hd))
            cross_out = custom_sdpa(q_ac, k_txt, v_txt).flatten(2)  # [B, L_a, dv]
            h_a = h_a + self.action_cross_to_out(cross_out)          # [B, L_a, da]

            # ── 3. Feed-forward (separate experts) ───────────────────────────
            # Video FFN
            h_v_n3 = (self.norm3(h_v.float()) * (1. + c_sc_v) + c_sh_v).type_as(h_v)
            h_v = (h_v.float() + self.ffn(h_v_n3).float() * c_g_v).type_as(h_v)

            # Action FFN
            h_a_n3 = (self.action_norm3(h_a.float()) * (1. + c_sc_a) + c_sh_a).type_as(h_a)
            h_a = (h_a.float() + self.action_ffn(h_a_n3).float() * c_g_a).type_as(h_a)

            return h_v, h_a

        # ── Base / inference path (single stream at dv) ──────────────────
        sst = self.scale_shift_table[None] + temb.float()
        sh, sc, g, c_sh, c_sc, c_g = [x.squeeze(1) for x in
            rearrange(sst, 'b l n c -> b n l c').chunk(6, dim=1)]

        norm_h = (self.norm1(hidden_states.float()) * (1. + sc) + sh).type_as(hidden_states)
        attn_out = self.attn1(norm_h, norm_h, norm_h, rotary_emb,
                              update_cache=update_cache, cache_name=cache_name)
        hidden_states = (hidden_states.float() + attn_out * g).type_as(hidden_states)

        norm_h = self.norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + self.attn2(norm_h, encoder_hidden_states,
                                                   encoder_hidden_states, None,
                                                   update_cache=0, cache_name=cache_name)

        norm_h = (self.norm3(hidden_states.float()) * (1. + c_sc) + c_sh).type_as(hidden_states)
        hidden_states = (hidden_states.float() + self.ffn(norm_h).float() * c_g).type_as(hidden_states)

        return hidden_states, None


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        "patch_embedding_mlp",
                                        "condition_embedder",
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder",
                             "scale_shift_table",
                             "action_scale_shift_table_final",
                             "norm1",
                             'action_norm1',
                             'text_norm1',
                             "norm2",
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3',
                             "norm_out",
                             "action_norm_out",
                             ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

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
                 attn_mode="torch",
                 action_hidden_dim: int = 768):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim   # dv = 3072
        da = action_hidden_dim                                  # da = 768
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)

        # Action stream embedder: 30-dim raw action → 768-dim action stream
        self.action_embedder = nn.Linear(action_dim, da)

        # Video condition embedder (at dv=3072)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        # Action condition embedder (at da=768, NOT a copy of video embedder)
        self.condition_embedder_action = WanTimeTextImageEmbedding(
            dim=da,
            time_freq_dim=freq_dim,
            time_proj_dim=da * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode,
                                action_hidden_dim=da) for _ in range(num_layers)
        ])

        # Video output head (at dv=3072)
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        # Action output head (at da=768)
        self.action_norm_out = FP32LayerNorm(da, eps, elementwise_affine=False)
        self.action_proj_out = nn.Linear(da, action_dim)
        self.action_scale_shift_table_final = nn.Parameter(
            torch.randn(1, 2, da) / da**0.5)

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
    
    def init_action_expert_from_video(self):
        """Copy-init Action Expert weights from Video Expert via dim interpolation + alpha scaling.

        Paper rule (Sec 3.3): alpha = sqrt(dv / da) = sqrt(3072/768) = 2.0
        Apply alpha ONLY when fan-in is compressed (input channels shrink).
        Never apply to biases or additive/modulation params (scale_shift_table).

        Mapping per block:
          action_to_q/k/v.weight  [dv,da]  ← interp to_q/k/v.weight [dv,dv] dim=1, ×alpha
          action_to_q/k/v.bias    [dv]     ← copy (same size)
          action_to_out.weight    [da,dv]  ← interp to_out[0].weight [dv,dv] dim=0, no alpha
          action_to_out.bias      [da]     ← interp to_out[0].bias [dv] dim=0, no alpha
          action_cross_to_q same rules as action_to_q (source: attn2.to_q)
          action_cross_to_out same rules as action_to_out (source: attn2.to_out[0])
          action_ffn.net[0].proj.weight [da_ffn,da] ← interp [ffn_dim,dv] dim0+dim1, ×alpha
          action_ffn.net[0].proj.bias   [da_ffn]    ← interp [ffn_dim] dim=0, no alpha
          action_ffn.net[2].weight      [da,da_ffn] ← interp [dv,ffn_dim] dim0+dim1, ×alpha
          action_ffn.net[2].bias        [da]         ← interp [dv] dim=0, no alpha
          action_scale_shift_table      [1,6,da]    ← interp [1,6,dv] dim=2, no alpha
        """
        dv = self.num_attention_heads * self.attention_head_dim  # 3072
        da = self.config.action_hidden_dim                        # 768
        alpha = math.sqrt(dv / da)  # = 2.0

        def interp_dim(w, new_len, dim):
            """Linear interp of one axis of an arbitrary-rank float tensor."""
            ndim = w.dim()
            # Permute target dim to the last position
            perm = list(range(ndim))
            perm.pop(dim)
            perm.append(dim)
            inv_perm = [perm.index(i) for i in range(ndim)]
            w = w.permute(*perm)                                     # [..., old_len]
            old_shape = w.shape
            w_flat = w.reshape(-1, old_shape[-1])                    # [N, old_len]
            w_out = F.interpolate(w_flat.unsqueeze(1).float(),
                                  size=new_len, mode='linear',
                                  align_corners=False).squeeze(1)    # [N, new_len]
            w = w_out.reshape(*old_shape[:-1], new_len)
            return w.permute(*inv_perm)

        with torch.no_grad():
            for block in self.blocks:
                # ── Self-attention: action_to_q/k/v, action_to_out ───────────
                for src_name, dst_name in [
                    ('attn1.to_q', 'action_to_q'),
                    ('attn1.to_k', 'action_to_k'),
                    ('attn1.to_v', 'action_to_v'),
                ]:
                    # Navigate dotted attr path
                    src = block
                    for part in src_name.split('.'): src = getattr(src, part)
                    dst = getattr(block, dst_name)
                    # weight [dv, dv] → [dv, da]: compress input dim=1, ×alpha
                    dst.weight.copy_(interp_dim(src.weight.float(), da, dim=1) * alpha)
                    if src.bias is not None and dst.bias is not None:
                        dst.bias.copy_(src.bias.float())   # [dv] → [dv], same shape

                # action_to_out: weight [dv,dv]→[da,dv] compress output dim=0, no alpha
                src_out = block.attn1.to_out[0]
                dst_out = block.action_to_out
                dst_out.weight.copy_(interp_dim(src_out.weight.float(), da, dim=0))
                if src_out.bias is not None and dst_out.bias is not None:
                    dst_out.bias.copy_(interp_dim(src_out.bias.float(), da, dim=0))

                # ── Cross-attention: action_cross_to_q, action_cross_to_out ──
                src_xq = block.attn2.to_q
                dst_xq = block.action_cross_to_q
                dst_xq.weight.copy_(interp_dim(src_xq.weight.float(), da, dim=1) * alpha)
                if src_xq.bias is not None and dst_xq.bias is not None:
                    dst_xq.bias.copy_(src_xq.bias.float())

                src_xout = block.attn2.to_out[0]
                dst_xout = block.action_cross_to_out
                dst_xout.weight.copy_(interp_dim(src_xout.weight.float(), da, dim=0))
                if src_xout.bias is not None and dst_xout.bias is not None:
                    dst_xout.bias.copy_(interp_dim(src_xout.bias.float(), da, dim=0))

                # ── Action FFN ────────────────────────────────────────────────
                v_ffn = block.ffn
                a_ffn = block.action_ffn
                da_ffn = a_ffn.net[0].proj.weight.shape[0]  # 3584

                # net[0].proj.weight: [ffn_dim=14336, dv=3072] → [da_ffn=3584, da=768]
                # compress dim=0 (14336→3584) then dim=1 (3072→768), ×alpha (fan-in compressed)
                w = v_ffn.net[0].proj.weight.float()
                w = interp_dim(w, da_ffn, dim=0)   # → [3584, 3072]
                w = interp_dim(w, da,     dim=1)   # → [3584, 768]
                a_ffn.net[0].proj.weight.copy_(w * alpha)

                if v_ffn.net[0].proj.bias is not None and a_ffn.net[0].proj.bias is not None:
                    a_ffn.net[0].proj.bias.copy_(
                        interp_dim(v_ffn.net[0].proj.bias.float(), da_ffn, dim=0))

                # net[2].weight: [dv=3072, ffn_dim=14336] → [da=768, da_ffn=3584]
                # compress dim=0 (3072→768) then dim=1 (14336→3584), ×alpha (fan-in compressed)
                w2 = v_ffn.net[2].weight.float()
                w2 = interp_dim(w2, da,     dim=0)   # → [768, 14336]
                w2 = interp_dim(w2, da_ffn, dim=1)   # → [768, 3584]
                a_ffn.net[2].weight.copy_(w2 * alpha)

                if v_ffn.net[2].bias is not None and a_ffn.net[2].bias is not None:
                    a_ffn.net[2].bias.copy_(
                        interp_dim(v_ffn.net[2].bias.float(), da, dim=0))

                # ── action_scale_shift_table: [1,6,dv=3072] → [1,6,da=768], no alpha ──
                block.action_scale_shift_table.copy_(
                    interp_dim(block.scale_shift_table.float(), da, dim=2))

        # ── condition_embedder_action: copy-init from condition_embedder ────────
        # Each Linear in the action embedder has input/output dims scaled by da/dv.
        # Apply alpha only when the action layer's fan-in is compressed vs the video layer.
        #
        # Layer mapping (src → dst, shape change, alpha?):
        #  time_embedder.linear_1  [3072,256]→[768,256]    dim=0 compress, fan-in 256=256, NO alpha
        #  time_embedder.linear_2  [3072,3072]→[768,768]   dim=0+dim=1,    fan-in 3072→768, ×alpha
        #  time_proj               [18432,3072]→[4608,768] dim=0+dim=1,    fan-in 3072→768, ×alpha
        #  text_embedder.linear_1  [3072,4096]→[768,4096]  dim=0 compress, fan-in 4096=4096, NO alpha
        #  text_embedder.linear_2  [3072,3072]→[768,768]   dim=0+dim=1,    fan-in 3072→768, ×alpha
        src_emb = self.condition_embedder
        dst_emb = self.condition_embedder_action
        time_proj_out_a = dst_emb.time_proj.weight.shape[0]  # da*6 = 4608

        with torch.no_grad():
            # time_embedder.linear_1: [dv,256]→[da,256], output-only compress, no alpha
            w = src_emb.time_embedder.linear_1.weight.float()   # [3072, 256]
            dst_emb.time_embedder.linear_1.weight.copy_(interp_dim(w, da, dim=0))
            dst_emb.time_embedder.linear_1.bias.copy_(
                interp_dim(src_emb.time_embedder.linear_1.bias.float(), da, dim=0))

            # time_embedder.linear_2: [dv,dv]→[da,da], fan-in dv→da compressed, ×alpha
            w = src_emb.time_embedder.linear_2.weight.float()   # [3072, 3072]
            w = interp_dim(w, da, dim=0)   # [768, 3072]
            w = interp_dim(w, da, dim=1)   # [768, 768]
            dst_emb.time_embedder.linear_2.weight.copy_(w * alpha)
            dst_emb.time_embedder.linear_2.bias.copy_(
                interp_dim(src_emb.time_embedder.linear_2.bias.float(), da, dim=0))

            # time_proj: [dv*6, dv]→[da*6, da], fan-in dv→da compressed, ×alpha
            w = src_emb.time_proj.weight.float()                # [18432, 3072]
            w = interp_dim(w, time_proj_out_a, dim=0)           # [4608, 3072]
            w = interp_dim(w, da, dim=1)                        # [4608, 768]
            dst_emb.time_proj.weight.copy_(w * alpha)
            dst_emb.time_proj.bias.copy_(
                interp_dim(src_emb.time_proj.bias.float(), time_proj_out_a, dim=0))

            # text_embedder.linear_1: [dv,4096]→[da,4096], fan-in 4096=4096, no alpha
            w = src_emb.text_embedder.linear_1.weight.float()   # [3072, 4096]
            dst_emb.text_embedder.linear_1.weight.copy_(interp_dim(w, da, dim=0))
            dst_emb.text_embedder.linear_1.bias.copy_(
                interp_dim(src_emb.text_embedder.linear_1.bias.float(), da, dim=0))

            # text_embedder.linear_2: [dv,dv]→[da,da], fan-in dv→da compressed, ×alpha
            w = src_emb.text_embedder.linear_2.weight.float()   # [3072, 3072]
            w = interp_dim(w, da, dim=0)   # [768, 3072]
            w = interp_dim(w, da, dim=1)   # [768, 768]
            dst_emb.text_embedder.linear_2.weight.copy_(w * alpha)
            dst_emb.text_embedder.linear_2.bias.copy_(
                interp_dim(src_emb.text_embedder.linear_2.bias.float(), da, dim=0))

        # ── action_scale_shift_table_final: [1,2,dv]→[1,2,da], no alpha ────────
        with torch.no_grad():
            self.action_scale_shift_table_final.copy_(
                interp_dim(self.scale_shift_table.float(), da, dim=2))

        v_ffn = self.blocks[0].ffn
        da_ffn = self.blocks[0].action_ffn.net[0].proj.weight.shape[0]
        print(f"init_action_expert_from_video: alpha={alpha:.4f} (sqrt({dv}/{da}))")
        print(f"  da={da}, da_ffn={da_ffn}, dv={dv}, ffn_dim={v_ffn.net[0].proj.weight.shape[0]}")
        print(f"  condition_embedder_action initialized from condition_embedder")

    def _input_embed(self, latents, input_type='latent'):
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)  # → [B, L, dv=3072]
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)       # → [B, L, da=768]
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def forward_train(self, input_dict):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent']        = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent']        = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size  = latent_dict['noisy_latents'].shape[0]

        # ── Embed inputs ──────────────────────────────────────────────────────
        # Video stream: [B, L_v_noisy, dv] each, flattened to [1, B*L_v_noisy, dv]
        lat_hs       = self._input_embed(latent_dict['noisy_latents'], 'latent').flatten(0, 1)[None]
        cond_lat_hs  = self._input_embed(latent_dict['latent'],        'latent').flatten(0, 1)[None]
        text_hs      = self._input_embed(latent_dict['text_emb'],      'text').flatten(0, 1)[None]

        # Action stream: [B, L_a_noisy, da=768] each, flattened to [1, B*L_a_noisy, da]
        act_hs       = self._input_embed(action_dict['noisy_latents'], 'action').flatten(0, 1)[None]
        cond_act_hs  = self._input_embed(action_dict['latent'],        'action').flatten(0, 1)[None]

        # Combined streams (separate tensors at different dims)
        h_v = torch.cat([lat_hs, cond_lat_hs], dim=1)    # [1, L_v, dv=3072]
        h_a = torch.cat([act_hs, cond_act_hs], dim=1)    # [1, L_a, da=768]

        L_v = h_v.shape[1]
        L_a = h_a.shape[1]

        # ── RoPE: separate grid_ids, combined for joint self-attention ────────
        lat_grid  = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]   # [1, 3, B*L_v_noisy]
        act_grid  = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]   # [1, 3, B*L_a_noisy]
        v_grid    = torch.cat([lat_grid] * 2, dim=2)                           # noisy + cond
        a_grid    = torch.cat([act_grid] * 2, dim=2)
        # Combined positional encodings for joint attention [L_v + L_a]
        rotary_emb = self.rope(torch.cat([v_grid, a_grid], dim=2))[:, :, None]  # [1, L_v+L_a, 1, C]

        # ── Time embeddings ───────────────────────────────────────────────────
        lat_ts = torch.cat([latent_dict['timesteps'].flatten(0, 1),
                            latent_dict['cond_timesteps'].flatten(0, 1)])[None]
        act_ts = torch.cat([action_dict['timesteps'].flatten(0, 1),
                            action_dict['cond_timesteps'].flatten(0, 1)])[None]

        H_lat, W_lat = latent_dict['noisy_latents'].shape[-2:]
        H_act, W_act = action_dict['noisy_latents'].shape[-2:]

        latent_temb, latent_ts_proj = self._time_embed(lat_ts, H_lat, W_lat,
                                                        dtype=h_v.dtype, action_mode=False)
        # latent_temb:    [1, L_v, dv=3072]   (used for final video output norm)
        # latent_ts_proj: [1, L_v, 6, dv]     (block AdaLN)

        action_temb, action_ts_proj = self._time_embed(act_ts, H_act, W_act,
                                                        dtype=h_a.dtype, action_mode=True)
        # action_temb:    [1, L_a, da=768]     (used for final action output norm)
        # action_ts_proj: [1, L_a, 6, da=768]  (block action AdaLN)

        # ── Block loop: two-stream MoT ────────────────────────────────────────
        for block in self.blocks:
            h_v, h_a = block(h_v, text_hs, latent_ts_proj, rotary_emb,
                             update_cache=False,
                             action_states=h_a, temb_action=action_ts_proj)

        # ── Video output norm + projection ────────────────────────────────────
        sst_v = self.scale_shift_table[None] + latent_temb[:, :, None, ...]  # [1, L_v, 2, dv]
        sh_v, sc_v = [x.squeeze(1) for x in
                      rearrange(sst_v, 'b l n c -> b n l c').chunk(2, dim=1)]
        h_v = (self.norm_out(h_v.float()) * (1. + sc_v) + sh_v).type_as(h_v)

        L_v_noisy = lat_hs.shape[1]
        latent_pred = self.proj_out(h_v[:, :L_v_noisy])
        latent_pred = rearrange(latent_pred, '1 (b l) (n c) -> b (l n) c',
                                n=math.prod(self.patch_size), b=batch_size)

        # ── Action output norm + projection ───────────────────────────────────
        sst_a = self.action_scale_shift_table_final[None] + action_temb[:, :, None, ...]  # [1, L_a, 2, da]
        sh_a, sc_a = [x.squeeze(1) for x in
                      rearrange(sst_a, 'b l n c -> b n l c').chunk(2, dim=1)]
        h_a = (self.action_norm_out(h_a.float()) * (1. + sc_a) + sh_a).type_as(h_a)

        L_a_noisy = act_hs.shape[1]
        action_pred = self.action_proj_out(h_a[:, :L_a_noisy])
        action_pred = rearrange(action_pred, '1 (b l) c -> b l c', b=batch_size)

        return latent_pred, action_pred

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        for block in self.blocks:
            latent_hidden_states, _ = block(latent_hidden_states,
                                            text_hidden_states,
                                            timestep_proj,
                                            rotary_emb,
                                            update_cache=update_cache,
                                            cache_name=cache_name,
                                            action_only=action_mode)

        if action_mode:
            # Action output norm + head at da=768
            sst_a = self.action_scale_shift_table_final[None] + temb[:, :, None, ...]
            shift_a, scale_a = rearrange(sst_a, 'b l n c -> b n l c').chunk(2, dim=1)
            shift_a = shift_a.squeeze(1)
            scale_a = scale_a.squeeze(1)
            latent_hidden_states = (self.action_norm_out(latent_hidden_states.float()) *
                                    (1. + scale_a) + shift_a).type_as(latent_hidden_states)
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            # Video output norm + head at dv=3072
            temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
            shift, scale = rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(2, dim=1)
            shift = shift.squeeze(1)
            scale = scale.squeeze(1)
            latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                    (1. + scale) + shift).type_as(latent_hidden_states)
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))

        return latent_hidden_states


if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
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
                                  attn_mode="torch")
    print(model)
