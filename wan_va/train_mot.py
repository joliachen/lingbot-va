# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
MoT fine-tuning entry point.

Requires a pre-converted WanMoTTransformer3DModel checkpoint (produced by
robot-icl/convert_to_mot.py) — the action expert is never copy-inited here.
If the checkpoint is missing, convert first:

    sbatch /scratch/zc2745/robot-icl/scripts/convert_to_mot.slurm

Differences from train.py:
  - loads the MoT checkpoint (--mot-checkpoint) instead of the base model,
    and validates that it actually is a WanMoTTransformer3DModel
  - freeze_backbone uses the current model_mot.py parameter names
    (action_attn1 / action_attn2 / ...); train.py's list predates the
    no-bottleneck rewrite and silently leaves the action attention frozen

Usage:
    torchrun --nproc_per_node=1 wan_va/train_mot.py \\
        --config-name libero_goal_object_train \\
        --mot-checkpoint /scratch/zc2745/robot-icl/checkpoints/lingbot-va-mot \\
        --save-root /scratch/zc2745/robot-icl/runs/mot_ft
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from utils import init_logger, logger, warmup_constant_lambda
from train import Trainer

# Parameter-name fragments of the MoT action expert (current model_mot.py):
#   per block: action_attn1.*, action_attn2.*, action_norm2.*, action_ffn.*,
#              action_scale_shift_table
#   top-level: action_scale_shift_table_final, action_embedder,
#              action_proj_out, condition_embedder_action.*
ACTION_EXPERT_PARAM_NAMES = (
    'action_attn1',
    'action_attn2',
    'action_norm2',
    'action_ffn',
    'action_scale_shift_table',   # also matches action_scale_shift_table_final
    'action_embedder',
    'action_proj_out',
    'condition_embedder_action',
)


def freeze_backbone_(transformer):
    """Freeze everything except the action expert. Returns trainable count."""
    transformer.requires_grad_(False)
    for name, param in transformer.named_parameters():
        if any(seg in name for seg in ACTION_EXPERT_PARAM_NAMES):
            param.requires_grad_(True)
    n_train = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in transformer.parameters())
    if n_train == 0:
        raise RuntimeError(
            'freeze_backbone_ matched no parameters — ACTION_EXPERT_PARAM_NAMES '
            'is out of sync with model_mot.py')
    logger.info(f"Freezing backbone; trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params")
    return n_train


def _validate_mot_checkpoint(path: str) -> None:
    cfg_file = Path(path) / 'transformer' / 'config.json'
    if not cfg_file.exists():
        raise FileNotFoundError(
            f'No MoT checkpoint at {path} — convert the base weights first:\n'
            f'    sbatch /scratch/zc2745/robot-icl/scripts/convert_to_mot.slurm')
    with open(cfg_file) as f:
        class_name = json.load(f).get('_class_name', '')
    if class_name != 'WanMoTTransformer3DModel':
        raise ValueError(
            f'{cfg_file} has _class_name={class_name!r}, expected '
            f"'WanMoTTransformer3DModel' — this is not a converted MoT checkpoint")


class MoTTrainer(Trainer):
    """Trainer that loads a pre-converted MoT checkpoint and freezes the
    backbone with the correct (current) action-expert parameter names."""

    def __init__(self, config):
        is_resume = hasattr(config, 'resume_from') and config.resume_from
        if not is_resume:
            _validate_mot_checkpoint(config.mot_checkpoint)
            config.wan22_pretrained_model_name_or_path = str(config.mot_checkpoint)

        # Bypass train.py's freeze block (its name list predates the current
        # model_mot.py and misses action_attn1/action_attn2); re-freeze below.
        freeze_requested = getattr(config, 'freeze_backbone', False)
        config.freeze_backbone = False
        super().__init__(config)
        config.freeze_backbone = freeze_requested

        if freeze_requested:
            freeze_backbone_(self.transformer)
            # Rebuild optimizer/scheduler over the trainable subset only
            # (the ones from super().__init__ cover all parameters).
            self.optimizer = torch.optim.AdamW(
                [p for p in self.transformer.parameters() if p.requires_grad],
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=1e-8,
                weight_decay=config.weight_decay,
                fused=True,
                foreach=False,
            )
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: warmup_constant_lambda(
                    step, warmup_steps=config.warmup_steps))


def run(args):
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.mot_checkpoint = args.mot_checkpoint

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, 'empty_emb.pt')

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"MoT checkpoint: {config.mot_checkpoint}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = MoTTrainer(config)
    trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune the MoT action expert")
    parser.add_argument(
        "--config-name",
        type=str,
        default='libero_goal_object_train',
        help="Config name",
    )
    parser.add_argument(
        "--mot-checkpoint",
        type=str,
        default='/scratch/zc2745/robot-icl/checkpoints/lingbot-va-mot',
        help="Pre-converted WanMoTTransformer3DModel checkpoint directory",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override config num_steps",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Override config save_interval",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override config dataset_path (and empty_emb_path)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
