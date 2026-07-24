# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Stage B: MoT post-train on LIBERO-90 with prompt (paraphrase) augmentation.
# Start weights: checkpoints/lingbot-va-mot (NOT the Stage-A output).
# Output packaged as: checkpoints/lingbot-va-mot-posttrain-libero-90-prompt-aug
# Requires the text_emb_variants sampling patch in lerobot_latent_dataset.py.
from easydict import EasyDict
from .va_libero_goal_object_cfg import va_libero_goal_object_cfg
import os

va_libero_90_prompt_aug_mot_train_cfg = EasyDict(
    __name__='Config: VA LIBERO-90 prompt-aug MoT posttrain')
va_libero_90_prompt_aug_mot_train_cfg.update(va_libero_goal_object_cfg)

va_libero_90_prompt_aug_mot_train_cfg.dataset_path = \
    '/scratch/zc2745/robot-icl/data/libero_90_prompt_aug_lerobot'
va_libero_90_prompt_aug_mot_train_cfg.empty_emb_path = os.path.join(
    va_libero_90_prompt_aug_mot_train_cfg.dataset_path, 'empty_emb.pt')

va_libero_90_prompt_aug_mot_train_cfg.enable_wandb  = False
va_libero_90_prompt_aug_mot_train_cfg.load_worker   = 4
va_libero_90_prompt_aug_mot_train_cfg.save_interval = 500
va_libero_90_prompt_aug_mot_train_cfg.gc_interval   = 10

# text dropout MUST equal pretraining value (paper §4.2: 0.1) — do not change
va_libero_90_prompt_aug_mot_train_cfg.cfg_prob = 0.1

# paper §4.3.2 hyperparams; 9× more episodes than one suite (4500 vs 500),
# so extend steps beyond the per-suite 4K.
va_libero_90_prompt_aug_mot_train_cfg.learning_rate  = 1e-5
va_libero_90_prompt_aug_mot_train_cfg.beta1          = 0.9
va_libero_90_prompt_aug_mot_train_cfg.beta2          = 0.95
va_libero_90_prompt_aug_mot_train_cfg.weight_decay   = 0.1
va_libero_90_prompt_aug_mot_train_cfg.warmup_steps   = 50
va_libero_90_prompt_aug_mot_train_cfg.batch_size     = 1
# Run with 4 GPUs (FSDP): effective batch = 4 ranks x 1 x 3 accum = 12
va_libero_90_prompt_aug_mot_train_cfg.gradient_accumulation_steps = 3
va_libero_90_prompt_aug_mot_train_cfg.num_steps      = 6000

# FULL-parameter fine-tune (paper-style posttrain): all 10.0B trainable.
# Requires >= 4x H200 with FSDP (see Stage-A config note).
va_libero_90_prompt_aug_mot_train_cfg.freeze_backbone = False
