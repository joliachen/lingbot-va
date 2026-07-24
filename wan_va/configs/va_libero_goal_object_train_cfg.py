# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_goal_object_cfg import va_libero_goal_object_cfg
import os

va_libero_goal_object_train_cfg = EasyDict(__name__='Config: VA LIBERO Goal+Object train')
va_libero_goal_object_train_cfg.update(va_libero_goal_object_cfg)

va_libero_goal_object_train_cfg.dataset_path = \
    '/scratch/zc2745/robot-icl/data/libero_goal_object_lerobot'
va_libero_goal_object_train_cfg.empty_emb_path = os.path.join(
    va_libero_goal_object_train_cfg.dataset_path, 'empty_emb.pt')

va_libero_goal_object_train_cfg.enable_wandb  = False
va_libero_goal_object_train_cfg.load_worker   = 4
va_libero_goal_object_train_cfg.save_interval = 250
va_libero_goal_object_train_cfg.gc_interval   = 10

# text dropout MUST equal pretraining value (0.1) — do not change
va_libero_goal_object_train_cfg.cfg_prob = 0.1

# Training hyperparameters from paper Section 4.3.2 / planning_finetune.md
va_libero_goal_object_train_cfg.learning_rate  = 1e-5
va_libero_goal_object_train_cfg.beta1          = 0.9
va_libero_goal_object_train_cfg.beta2          = 0.95
va_libero_goal_object_train_cfg.weight_decay   = 0.1
va_libero_goal_object_train_cfg.warmup_steps   = 50
va_libero_goal_object_train_cfg.batch_size     = 1
va_libero_goal_object_train_cfg.gradient_accumulation_steps = 4
va_libero_goal_object_train_cfg.num_steps      = 2000
va_libero_goal_object_train_cfg.sequence_length = 100000

# Freeze video backbone; train the full action stream (action_attn1/2, action_ffn,
# norms, modulation tables, embedder/head) — 5.0B of 10.0B params. AdamW states
# for 5B params need ~40 GB fp32: plan for an H200-class GPU.
va_libero_goal_object_train_cfg.freeze_backbone = True
