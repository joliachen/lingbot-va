# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    blocks = model.mot_blocks if hasattr(model, 'mot_blocks') else model.blocks
    for layer_id, transformer_block in enumerate(blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        if hasattr(block, 'forward_joint'):
            # WanMoTTransformerBlock: forward_joint reads all four attention
            # modules' sub-layers directly (to_q/to_k/to_v/to_out), so their
            # params must live in the block's own FSDP group — a separate
            # fully_shard group would never unshard (no module call). Only
            # module-called children (the FFNs) get their own groups.
            fully_shard(block.ffn, **fsdp_config)
            fully_shard(block.action_ffn, **fsdp_config)
            fully_shard(block, **fsdp_config)
        else:
            # WanTransformer3DModel (shared backbone)
            fully_shard(block.attn1, **fsdp_config)
            fully_shard(block.attn2, **fsdp_config)
            fully_shard(block.ffn, **fsdp_config)
            fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
