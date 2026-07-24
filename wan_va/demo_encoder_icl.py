# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Offline demo KV encoding for Video ICL (planning_icl.md section 2.1).

Encodes ONE successful LIBERO demo (video only, no action tokens) through the
MoT video stream exactly the way the eval server encodes real rollout history
(`VA_Server._compute_kv_cache`, video branch): chunked streaming VAE encode,
timestep-0 conditioning forwards, causal KV-cache pool — but into a dedicated
'demo' cache pool sized exactly so nothing is ever evicted. The pool contents
(post-RoPE K/V per layer, temporal order) are then exported as the demo KV
pack that wan_va_server_icl.py injects at inference time.

Frame chunking mirrors the closed-loop client cadence exactly:
    init      : frame 0 alone                -> 1 latent frame
    chunk 0   : frames 1..12   (12 frames)   -> 3 latent frames (+init = 4)
    chunk k>0 : frames 13+16(k-1)..12+16k    -> 4 latent frames
so demo chunk boundaries line up with rollout chunk boundaries and
chunk_end_steps[k] gives the env-step index a GT-progress matcher needs.

Usage (1 GPU):
    torchrun --nproc_per_node 1 wan_va/demo_encoder_icl.py \
        --config-name libero_long \
        --checkpoint-path /scratch/zc2745/robot-icl/checkpoints/lingbot-va-mot \
        --hdf5 /scratch/zc2745/robot-icl/data/libero/libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5 \
        --demo-key demo_0 \
        --out /scratch/zc2745/robot-icl/data/icl_demo_packs/libero10_task0_demo0.pt
"""
import argparse
import os
import re
import sys

import h5py
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.util import init_distributed
from modules.model_mot_icl import export_demo_kv
from utils import init_logger, logger
from wan_va_server import VA_Server

DEMO_CACHE = 'demo'


def task_prompt_from_filename(stem: str) -> str:
    m = re.match(r'^(?:[A-Z0-9_]*?SCENE\d+_)?(.*?)(?:_demo)?$', stem)
    return m.group(1).replace('_', ' ').strip().lower()


def load_demo_frames(hdf5_path, demo_key, cam_keys):
    """Returns frames: list[dict cam_key -> HxWx3 uint8 (already flipped)], states."""
    with h5py.File(hdf5_path, 'r') as f:
        demo = f['data'][demo_key]
        cams = {}
        cams['observation.images.agentview_rgb'] = np.asarray(
            demo['obs']['agentview_rgb'])[:, ::-1].copy()
        cams['observation.images.eye_in_hand_rgb'] = np.asarray(
            demo['obs']['eye_in_hand_rgb'])[:, ::-1].copy()
        states = np.asarray(demo['states'], dtype=np.float32)
    T = states.shape[0]
    frames = [{k: cams[k][t] for k in cam_keys} for t in range(T)]
    return frames, states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config-name', default='libero_long')
    ap.add_argument('--checkpoint-path', required=True,
                    help='dir with transformer/ vae/ text_encoder/ tokenizer/ — '
                         'MUST be the same model the ICL server will run')
    ap.add_argument('--hdf5', required=True)
    ap.add_argument('--demo-key', default='demo_0')
    ap.add_argument('--out', required=True)
    ap.add_argument('--prompt', default=None,
                    help='override task prompt (default: derived from filename)')
    ap.add_argument('--max-chunks', type=int, default=0, help='0 = all full chunks')
    args = ap.parse_args()

    config = VA_CONFIGS[args.config_name]
    config.wan22_pretrained_model_name_or_path = args.checkpoint_path
    config.save_root = os.path.join(os.path.dirname(args.out), 'encoder_tmp')
    config.rank, config.local_rank, config.world_size = 0, 0, 1
    init_distributed(1, 0, 0)

    server = VA_Server(config)
    prompt = args.prompt or task_prompt_from_filename(
        os.path.basename(args.hdf5).rsplit('.', 1)[0])
    logger.info(f'demo prompt: "{prompt}"')
    server._reset(prompt=prompt)

    frames, states = load_demo_frames(args.hdf5, args.demo_key,
                                      config.obs_cam_keys)
    T = len(frames)
    n_chunks = 1 + (T - 13) // 16
    if args.max_chunks > 0:
        n_chunks = min(n_chunks, args.max_chunks)
    assert n_chunks >= 1, f'demo too short: {T} steps'
    chunk_end_steps = [12 + 16 * k for k in range(n_chunks)]

    frame_chunk_size = config.frame_chunk_size  # 4 latent frames per chunk
    tokens_per_chunk = (frame_chunk_size * server.latent_height *
                        server.latent_width) // int(np.prod(config.patch_size))
    total_tokens = n_chunks * tokens_per_chunk
    batch_size = 2 if server.use_cfg else 1
    logger.info(f'demo {args.demo_key}: T={T} steps -> {n_chunks} chunks, '
                f'{tokens_per_chunk} tok/chunk, batch={batch_size}')

    # exactly-sized pool: nothing ever evicted -> slots stay in temporal order
    for block in server.transformer.blocks:
        block.attn1.init_kv_cache(DEMO_CACHE, total_tokens,
                                  server.transformer.num_attention_heads,
                                  server.transformer.attention_head_dim,
                                  server.device, server.dtype, batch_size)

    latents_per_chunk = []
    with torch.no_grad():
        init_latent = server._encode_obs({'obs': [frames[0]]})
        for k in range(n_chunks):
            if k == 0:
                chunk_frames = frames[1:13]
            else:
                chunk_frames = frames[13 + 16 * (k - 1):13 + 16 * k]
            latent = server._encode_obs({'obs': chunk_frames})
            if k == 0:
                latent = torch.cat([init_latent, latent], dim=2)
            latents_per_chunk.append(latent.detach().cpu())
            input_dict = server._prepare_latent_input(
                latent.to(server.dtype), None,
                frame_st_id=k * frame_chunk_size)
            server.transformer(
                server._repeat_input_for_cfg(input_dict['latent_res_lst']),
                update_cache=2, cache_name=DEMO_CACHE, action_mode=False)
            logger.info(f'encoded demo chunk {k + 1}/{n_chunks}')

    k_list, v_list = export_demo_kv(server.transformer, DEMO_CACHE)
    assert k_list[0].shape[1] == total_tokens, (
        f'exported {k_list[0].shape[1]} tokens, expected {total_tokens}')

    pack = {
        'meta': {
            'hdf5': args.hdf5,
            'demo_key': args.demo_key,
            'prompt': prompt,
            'checkpoint_path': args.checkpoint_path,
            'n_chunks': n_chunks,
            'tokens_per_chunk': tokens_per_chunk,
            'frames_per_chunk': frame_chunk_size,
            'chunk_end_steps': chunk_end_steps,
            'batch_layout': ['cond', 'uncond'][:batch_size],
            'latent_height': server.latent_height,
            'latent_width': server.latent_width,
            'demo_len_steps': T,
        },
        'k': torch.stack(k_list),   # [L, B, N, H, D] post-RoPE, native positions
        'v': torch.stack(v_list),
        'demo_latents': torch.cat(latents_per_chunk, dim=2),  # [1,48,F,h,w]
        'demo_states': torch.from_numpy(states),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(pack, args.out)
    # small sidecar for the eval client (GT progress needs states only)
    np.savez(args.out.replace('.pt', '_states.npz'),
             states=states,
             chunk_end_steps=np.array(chunk_end_steps),
             n_chunks=n_chunks)
    logger.info(f'saved demo pack: {args.out} '
                f'(k {tuple(pack["k"].shape)}, v {tuple(pack["v"].shape)})')


if __name__ == '__main__':
    init_logger()
    main()
