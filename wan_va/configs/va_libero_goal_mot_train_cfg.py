# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Stage A: MoT post-train on LIBERO-Goal (original prompts, no paraphrase aug).
# Start weights: checkpoints/lingbot-va-mot (converted, verified).
# Output packaged as: checkpoints/lingbot-va-mot-posttrain-libero-goal
from easydict import EasyDict
from .va_libero_goal_object_cfg import va_libero_goal_object_cfg
import os

va_libero_goal_mot_train_cfg = EasyDict(__name__='Config: VA LIBERO-Goal MoT posttrain')
va_libero_goal_mot_train_cfg.update(va_libero_goal_object_cfg)

va_libero_goal_mot_train_cfg.dataset_path = \
    '/scratch/zc2745/robot-icl/data/libero_goal_lerobot'
va_libero_goal_mot_train_cfg.empty_emb_path = os.path.join(
    va_libero_goal_mot_train_cfg.dataset_path, 'empty_emb.pt')

va_libero_goal_mot_train_cfg.enable_wandb  = True    # ml-nyush/robot-icl (WANDB_* in ~/.bashrc)
va_libero_goal_mot_train_cfg.load_worker   = 4
va_libero_goal_mot_train_cfg.save_interval = 500
va_libero_goal_mot_train_cfg.gc_interval   = 10

# text dropout MUST equal pretraining value (paper §4.2: 0.1) — do not change
va_libero_goal_mot_train_cfg.cfg_prob = 0.1

# paper §4.3.2 (LIBERO posttrain): lr 1e-5, 4K steps;
# upstream va_libero_train_cfg: batch 1 × grad_accum 10, weight_decay 0.1
va_libero_goal_mot_train_cfg.learning_rate  = 1e-5
va_libero_goal_mot_train_cfg.beta1          = 0.9
va_libero_goal_mot_train_cfg.beta2          = 0.95
va_libero_goal_mot_train_cfg.weight_decay   = 0.1
va_libero_goal_mot_train_cfg.warmup_steps   = 50
va_libero_goal_mot_train_cfg.batch_size     = 1
# Run with 2 GPUs (FSDP): effective batch = 2 ranks x 1 x 5 accum = 10,
# matching upstream posttrain exactly. (2xH200 fits full-param training —
# ~80GB/GPU of fp32 state — and schedules far faster than 4-GPU asks;
# sanity 12980232 validated memory + loss + checkpoint save at 2 GPUs.)
# posttraining_fix.md phase 1: no episode packing exists (train.py asserts
# batch_size==1 per rank, dataset __getitem__ is one whole episode — see
# model_mot.py forward_train), so tokens/step was ~11.5K vs paper's ~100K.
# 43 x 2 ranks x ~1155 avg tokens/episode ~= 99K tokens/step to match paper
# without touching the packing-dataset/FlexAttn work (phase 2).
va_libero_goal_mot_train_cfg.gradient_accumulation_steps = 43
va_libero_goal_mot_train_cfg.num_steps      = 4000

# FULL-parameter fine-tune (paper-style posttrain): video stream + action
# stream + embedders/heads all trainable (10.0B). Does NOT fit on one GPU
# (fp32 params 40GB + AdamW 80GB) — requires >= 4x H200 with FSDP.
va_libero_goal_mot_train_cfg.freeze_backbone = False
