# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Single-task diagnostic: full-parameter MoT posttrain on ONLY "put the wine
# bottle on the rack" (task 9), the worst/near-worst performer across every
# LIBERO-goal eval so far (0% closed-loop in the baseline and both-streams
# LoRA runs). Isolates whether multi-task interference (diluting the 500-
# episode/10-task dataset) explains the poor score, vs. something more basic
# that single-task overfitting wouldn't fix either.
#
# Identical hyperparameters to va_libero_goal_mot_train_cfg (lr, steps,
# grad_accum, token budget) — the ONLY variable changed is task_filter, to
# keep this a clean single-task-vs-multi-task comparison.
#
# Start weights: checkpoints/lingbot-va-mot (converted, verified — same base
# as every other posttrain in this project, not the multi-task checkpoint).
# Output packaged as: checkpoints/lingbot-va-mot-posttrain-put-the-wine-bottle-on-the-rack
from easydict import EasyDict
from .va_libero_goal_mot_train_cfg import va_libero_goal_mot_train_cfg

va_libero_wine_bottle_rack_mot_train_cfg = EasyDict(__name__='Config: VA LIBERO wine-bottle-rack single-task MoT posttrain')
va_libero_wine_bottle_rack_mot_train_cfg.update(va_libero_goal_mot_train_cfg)

va_libero_wine_bottle_rack_mot_train_cfg.task_filter = 'put the wine bottle on the rack'
