# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg
from .va_libero_cfg import va_libero_cfg
from .va_libero_train_cfg import va_libero_train_cfg
from .va_libero_i2va import va_libero_i2va_cfg
from .va_libero_goal_object_cfg import va_libero_goal_object_cfg
from .va_libero_goal_object_train_cfg import va_libero_goal_object_train_cfg
from .va_libero_goal_mot_train_cfg import va_libero_goal_mot_train_cfg
from .va_libero_wine_bottle_rack_mot_train_cfg import va_libero_wine_bottle_rack_mot_train_cfg
from .va_libero_90_prompt_aug_mot_train_cfg import va_libero_90_prompt_aug_mot_train_cfg
from .va_libero_long_cfg import va_libero_long_cfg
from .va_libero_goal_shared_train_cfg import va_libero_goal_shared_train_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
    'libero': va_libero_cfg,
    'libero_train': va_libero_train_cfg,
    'libero_i2av': va_libero_i2va_cfg,
    'libero_goal_object': va_libero_goal_object_cfg,
    'libero_goal_object_train': va_libero_goal_object_train_cfg,
    'libero_goal_mot_train': va_libero_goal_mot_train_cfg,
    'libero_wine_bottle_rack_mot_train': va_libero_wine_bottle_rack_mot_train_cfg,
    'libero_90_prompt_aug_mot_train': va_libero_90_prompt_aug_mot_train_cfg,
    'libero_long': va_libero_long_cfg,
    'libero_goal_shared_train': va_libero_goal_shared_train_cfg,
}