# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Official (non-MoT, shared-backbone WanTransformer3DModel) posttrain checkpoint
# for LIBERO-Long (libero_10 benchmark) — reference-quality baseline, not ours.
# Inherits va_libero_cfg defaults as-is (matches how va_libero_i2va_cfg, the
# authors' own demo config for this same checkpoint, is built) rather than the
# libero_goal_object overrides, which were tuned for our MoT finetune.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg

va_libero_long_cfg = EasyDict(__name__='Config: VA LIBERO-Long (official posttrain)')
va_libero_long_cfg.update(va_libero_cfg)

va_libero_long_cfg.wan22_pretrained_model_name_or_path = \
    '/scratch/zc2745/robot-icl/checkpoints/lingbot-va-posttrain-libero-long'
va_libero_long_cfg.env_type = 'libero'
