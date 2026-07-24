# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg

va_libero_goal_object_cfg = EasyDict(__name__='Config: VA LIBERO Goal+Object')
va_libero_goal_object_cfg.update(va_libero_cfg)

# Converted MoT checkpoint (full layout: transformer/ + vae/ + text_encoder/ +
# tokenizer/). For eval of a fine-tuned run, point at a checkpoint dir with the
# same layout (train_mot saves transformer/ only — symlink the rest from here).
va_libero_goal_object_cfg.wan22_pretrained_model_name_or_path = \
    '/scratch/zc2745/robot-icl/checkpoints/lingbot-va-mot-posttrain-libero-goal-gradaccum43'

va_libero_goal_object_cfg.attn_window      = 72
va_libero_goal_object_cfg.frame_chunk_size = 4
va_libero_goal_object_cfg.env_type         = 'libero'
va_libero_goal_object_cfg.height           = 128
va_libero_goal_object_cfg.width            = 128
va_libero_goal_object_cfg.action_dim       = 30
va_libero_goal_object_cfg.action_per_frame = 4

# Both cameras are available (confirmed by HDF5 inspect)
va_libero_goal_object_cfg.obs_cam_keys = [
    'observation.images.agentview_rgb',
    'observation.images.eye_in_hand_rgb',
]

va_libero_goal_object_cfg.guidance_scale        = 5.0
va_libero_goal_object_cfg.action_guidance_scale = 1.0
va_libero_goal_object_cfg.num_inference_steps        = 25
va_libero_goal_object_cfg.action_num_inference_steps = 50
va_libero_goal_object_cfg.snr_shift        = 5.0
va_libero_goal_object_cfg.action_snr_shift = 0.05

# Use first 7 dims (native LIBERO action: xyz + euler + gripper)
va_libero_goal_object_cfg.used_action_channel_ids = list(range(7))
_inv = [7] * 30  # index 7 = the zero-padded slot in action_paded
for i, j in enumerate(va_libero_goal_object_cfg.used_action_channel_ids):
    _inv[j] = i
va_libero_goal_object_cfg.inverse_used_action_channel_ids = _inv

# Norm stats from LIBERO-Long (similar robot/action space; update after compute_norm_stats.py)
va_libero_goal_object_cfg.norm_stat = va_libero_cfg.norm_stat

# Enable MoT action expert FFN (see model.py changes)
va_libero_goal_object_cfg.use_mot_action_expert = True
