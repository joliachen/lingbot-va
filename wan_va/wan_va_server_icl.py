# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Video ICL inference server (planning_icl.md sections 2.2/2.3).

VA_Server_ICL = VA_Server + demo KV injection:

  * loads a demo KV pack produced by demo_encoder_icl.py and installs the
    ICL attention patch (modules/model_mot_icl.py) on the loaded transformer;
  * during `_infer` (all denoise forwards, video and action) the prediction
    tokens additionally attend demo_kv[:p_hat], RoPE-shifted to the time
    region before history;
  * during `_compute_kv_cache` ICL is disabled, so history KV is bit-exact
    with the non-ICL baseline (the plan's "history does not see demo" rule).

Progress modes (--icl-progress-mode):
  client    — E1: the eval client sends `demo_progress` (GT from privileged
              sim state, see evaluation/libero/client_icl.py) with each
              chunk-inference request.
  timestamp — E5: p_hat = rollout chunk index + 1 (hard time alignment).
  retrieval — E4 (Phase 2): latent-feature retrieval, not wired up yet.

Baseline note: with --icl-demo-pack absent this file refuses to run — use the
original wan_va_server.py for E0a/E0b; this keeps the baseline path untouched.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from modules.model_mot_icl import ICLState, install_icl
from utils import init_logger, logger, run_async_server_mode
from wan_va_server import VA_Server


class VA_Server_ICL(VA_Server):

    def __init__(self, job_config):
        super().__init__(job_config)
        pack = torch.load(job_config.icl_demo_pack, map_location='cpu',
                          weights_only=False)
        meta = pack['meta']
        if os.path.realpath(meta['checkpoint_path']) != os.path.realpath(
                job_config.wan22_pretrained_model_name_or_path):
            logger.warning(
                f"demo pack encoded with {meta['checkpoint_path']} but server "
                f"runs {job_config.wan22_pretrained_model_name_or_path} — the "
                "demo KV is only valid for the encoding model!")
        n_layers = pack['k'].shape[0]
        demo_k = [pack['k'][l].to(self.device, self.dtype) for l in range(n_layers)]
        demo_v = [pack['v'][l].to(self.device, self.dtype) for l in range(n_layers)]
        self.icl_meta = meta
        self.icl_state = ICLState(
            demo_k, demo_v,
            tokens_per_chunk=meta['tokens_per_chunk'],
            n_chunks=meta['n_chunks'],
            gap_frames=job_config.icl_gap_frames,
            frames_per_chunk=meta['frames_per_chunk'],
            capture_attn_mass=job_config.icl_log_attn_mass,
        )
        install_icl(self.transformer, self.icl_state)
        self.icl_lookahead = job_config.icl_lookahead_chunks
        self.icl_progress_mode = job_config.icl_progress_mode
        if self.icl_progress_mode == 'retrieval':
            raise NotImplementedError('retrieval progress is Phase 2 (E4)')
        self._client_p_hat = 1
        self._chunk_idx = 0
        logger.info(
            f"ICL ready: demo {meta['demo_key']} ({meta['n_chunks']} chunks, "
            f"prompt '{meta['prompt']}'), mode={self.icl_progress_mode}, "
            f"lookahead={self.icl_lookahead}, gap={job_config.icl_gap_frames}f")

    def _reset(self, prompt=None):
        super()._reset(prompt=prompt)
        self.icl_state.enabled = False
        self.icl_state.set_progress(0)
        self._client_p_hat = 1
        self._chunk_idx = 0
        if prompt is not None and prompt != self.icl_meta['prompt']:
            logger.warning(f"rollout prompt '{prompt}' != demo prompt "
                           f"'{self.icl_meta['prompt']}'")

    def infer(self, obs):
        if 'demo_progress' in obs:
            self._client_p_hat = int(obs['demo_progress'])
        return super().infer(obs)

    def _current_p_hat(self):
        if self.icl_progress_mode == 'timestamp':
            p = self._chunk_idx + 1
        else:  # 'client'
            p = self._client_p_hat
        return min(p + self.icl_lookahead, self.icl_state.n_chunks)

    def _infer(self, obs, frame_st_id=0):
        p_eff = self._current_p_hat()
        self.icl_state.set_progress(p_eff)
        self.icl_state.enabled = True
        self.icl_state.attn_mass = {'video': {}, 'action': {}}
        try:
            result = super()._infer(obs, frame_st_id=frame_st_id)
        finally:
            self.icl_state.enabled = False
        if self.icl_state.capture_attn_mass:
            self._log_attn_mass(p_eff)
        logger.info(f'ICL chunk {self._chunk_idx}: p_hat={p_eff}/'
                    f'{self.icl_state.n_chunks} '
                    f'({self.icl_state.active_tokens} demo tokens)')
        self._chunk_idx += 1
        return result

    def _compute_kv_cache(self, obs):
        assert not self.icl_state.enabled, 'ICL must be off for history commits'
        return super()._compute_kv_cache(obs)

    def _log_attn_mass(self, p_eff):
        rec = {'chunk': self._chunk_idx, 'p_hat': p_eff}
        for stream in ('video', 'action'):
            masses = self.icl_state.attn_mass[stream]
            if masses:
                vals = [masses[l] for l in sorted(masses)]
                rec[stream] = {'mean': float(np.mean(vals)),
                               'max': float(np.max(vals)),
                               'per_layer': [round(v, 5) for v in vals]}
        path = os.path.join(self.exp_save_root, 'icl_attn_mass.jsonl')
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        for stream in ('video', 'action'):
            if stream in rec:
                logger.info(f"ICL attn mass [{stream}] chunk {rec['chunk']}: "
                            f"mean={rec[stream]['mean']:.4f} "
                            f"max={rec[stream]['max']:.4f}")


def run(args):
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.checkpoint_path:
        config.wan22_pretrained_model_name_or_path = args.checkpoint_path
    if args.video_cfg_scale is not None:
        config.guidance_scale = args.video_cfg_scale
    if args.action_cfg_scale is not None:
        config.action_guidance_scale = args.action_cfg_scale
    config.icl_demo_pack = args.icl_demo_pack
    config.icl_gap_frames = args.icl_gap_frames
    config.icl_lookahead_chunks = args.icl_lookahead_chunks
    config.icl_progress_mode = args.icl_progress_mode
    config.icl_log_attn_mass = args.icl_log_attn_mass

    rank = int(os.getenv('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    init_distributed(world_size, local_rank, rank)
    config.rank, config.local_rank, config.world_size = rank, local_rank, world_size
    model = VA_Server_ICL(config)
    logger.info('****************************** ICL Server mode ******************************')
    run_async_server_mode(model, local_rank, config.host, port)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, default='libero_long')
    parser.add_argument('--port', type=int, default=None)
    parser.add_argument('--save_root', type=str, default=None)
    parser.add_argument('--checkpoint-path', type=str, default=None)
    parser.add_argument('--video-cfg-scale', type=float, default=None)
    parser.add_argument('--action-cfg-scale', type=float, default=None)
    parser.add_argument('--icl-demo-pack', type=str, required=True,
                        help='demo KV pack from demo_encoder_icl.py')
    parser.add_argument('--icl-gap-frames', type=int, default=4,
                        help='T_gap: latent frames between demo end and t=0')
    parser.add_argument('--icl-lookahead-chunks', type=int, default=0,
                        help="K': demo chunks beyond p_hat (E2 ablation)")
    parser.add_argument('--icl-progress-mode', type=str, default='client',
                        choices=['client', 'timestamp', 'retrieval'])
    parser.add_argument('--icl-log-attn-mass', action='store_true')
    args = parser.parse_args()
    run(args)
    logger.info('Finish all process!!!!!!!!!!!!')


if __name__ == '__main__':
    init_logger()
    main()
