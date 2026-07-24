# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Control experiment: posttrain the SHARED BACKBONE (lingbot-va-base,
# WanTransformer3DModel — the officially released/validated architecture)
# on LIBERO-Goal, to isolate whether the low MoT success rate comes from
# the MoT conversion/optimization or from the data/eval pipeline.
#
# Recipe mirrors the confirmed-working official LIBERO finetune:
#   AdamW (0.9, 0.95), lr 1e-5 constant, wd 0.1, warmup 10, clip 2.0,
#   batch 1/device, 4000 steps, per-suite training.
# Effective batch: 18 accum x 4 GPUs = 72 episodes/step (~83K tokens/step,
# paper scale; the working reference used 30 x 4 = 120 but that costs ~23h.
# 18 fits the 15h budget on 4xH100 at the measured ~0.72 s/episode).
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os

va_libero_goal_shared_train_cfg = EasyDict(
    __name__='Config: VA LIBERO-Goal shared-backbone posttrain')
va_libero_goal_shared_train_cfg.update(va_libero_cfg)
va_libero_goal_shared_train_cfg.__name__ = 'Config: VA LIBERO-Goal shared-backbone posttrain'

va_libero_goal_shared_train_cfg.wan22_pretrained_model_name_or_path = \
    '/scratch/zc2745/robot-icl/checkpoints/lingbot-va-base'

va_libero_goal_shared_train_cfg.dataset_path = \
    '/scratch/zc2745/robot-icl/data/libero_goal_lerobot'
va_libero_goal_shared_train_cfg.empty_emb_path = os.path.join(
    va_libero_goal_shared_train_cfg.dataset_path, 'empty_emb.pt')

va_libero_goal_shared_train_cfg.enable_wandb  = True
va_libero_goal_shared_train_cfg.load_worker   = 4
va_libero_goal_shared_train_cfg.save_interval = 500
va_libero_goal_shared_train_cfg.gc_interval   = 10

# text dropout, same as pretraining (paper §4.2)
va_libero_goal_shared_train_cfg.cfg_prob = 0.1

va_libero_goal_shared_train_cfg.learning_rate  = 1e-5
va_libero_goal_shared_train_cfg.beta1          = 0.9
va_libero_goal_shared_train_cfg.beta2          = 0.95
va_libero_goal_shared_train_cfg.weight_decay   = 0.1
va_libero_goal_shared_train_cfg.warmup_steps   = 10
va_libero_goal_shared_train_cfg.batch_size     = 1
va_libero_goal_shared_train_cfg.gradient_accumulation_steps = 18
va_libero_goal_shared_train_cfg.num_steps      = 4000

# Plain shared backbone: no MoT copy-init, no freezing.
va_libero_goal_shared_train_cfg.use_mot_action_expert = False
va_libero_goal_shared_train_cfg.freeze_backbone       = False
