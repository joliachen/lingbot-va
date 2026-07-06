# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    from diffusers import AutoencoderKLWan  # lazy: avoids flash_attn at import time
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    import json, os
    with open(os.path.join(transformer_path, "config.json")) as _f:
        _class_name = json.load(_f).get("_class_name", "WanTransformer3DModel")

    if _class_name == "WanMoTTransformer3DModel":
        from .model_mot import WanMoTTransformer3DModel
        model = WanMoTTransformer3DModel.from_pretrained(
            transformer_path, torch_dtype=torch_dtype, **kwargs)
    else:
        inferred_action_hidden_dim = _infer_action_hidden_dim(transformer_path)
        if inferred_action_hidden_dim is not None:
            kwargs.setdefault("action_hidden_dim", inferred_action_hidden_dim)
        model = WanTransformer3DModel.from_pretrained(
            transformer_path, torch_dtype=torch_dtype, **kwargs)

    # Materialize any meta tensors (e.g. newly-added action_ffn weights not in checkpoint).
    _materialize_meta_tensors(model, dtype=torch_dtype)
    return model.to(torch_device)


def _infer_action_hidden_dim(transformer_path):
    """Infer base action stream width from checkpoint weights when config omits it."""
    import json
    import os
    from glob import glob

    target_key = "action_embedder.bias"
    shard_names = []
    index_path = os.path.join(transformer_path,
                              "diffusion_pytorch_model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        shard_name = weight_map.get(target_key)
        if shard_name is not None:
            shard_names.append(os.path.join(transformer_path, shard_name))

    if not shard_names:
        shard_names = sorted(glob(os.path.join(transformer_path, "*.safetensors")))

    try:
        from safetensors import safe_open
    except Exception:
        return None

    for shard_path in shard_names:
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if target_key in f.keys():
                return f.get_tensor(target_key).shape[0]
    return None


def _materialize_meta_tensors(model: torch.nn.Module, dtype=torch.float32) -> None:
    """Replace meta-device parameters with zero-initialized tensors on CPU."""
    with torch.no_grad():
        for name, param in list(model.named_parameters()):
            if param.is_meta:
                parts = name.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                setattr(mod, parts[-1],
                        torch.nn.Parameter(torch.zeros(param.shape, dtype=dtype, device="cpu")))


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
