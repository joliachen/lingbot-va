"""
LIBERO eval client for Video ICL Phase 1 (E1): closed-loop rollout that sends
GROUND-TRUTH demo progress (from privileged mujoco sim state) to the ICL
server with every chunk-inference request.

Progress: before each chunk, `env.get_sim_state()` is matched against the
demo's per-step `states` array (same flattened-mujoco format as the
benchmark's init states) with a monotonic, range-bounded L2 argmin
(retrieval_icl.GTProgressMatcher); the matched env step is mapped to the
inclusive demo-chunk index p_hat that demo_encoder_icl.py's chunk boundaries
define. The p_hat trajectory is saved per episode for the alignment-error
diagnostic (plan section 5).

Everything else (env driving, KV-cache feedback, action clipping, video
saving) matches evaluation/libero/client.py closed-loop behavior exactly.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from libero.libero import benchmark
from tqdm import tqdm

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'wan_va'))

from client import construct_single_env, init_single_env, env_one_step, save_video
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
from retrieval_icl import GTProgressMatcher


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, matcher,
            open_loop=False):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
        'bddl_file_name': benchmark_instance.get_task_bddl_file_path(task_idx),
        'camera_heights': 128,
        'camera_widths': 128,
    }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    matcher.reset()
    model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    p_hat_traj = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        p_hat, matched_step = matcher.update(cur_env.get_sim_state())
        p_hat_traj.append({'env_timestep': int(cur_env.env.timestep),
                           'p_hat': int(p_hat),
                           'matched_demo_step': int(matched_step)})
        if open_loop and not first:
            # Blind continuation: no real obs; KV advance below uses imagine=True.
            ret = model.infer(dict(prompt=prompt, demo_progress=p_hat))
        else:
            ret = model.infer(dict(obs=first_obs, prompt=prompt, demo_progress=p_hat))
        action = np.clip(ret['action'], -1.0, 1.0)

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                observes, done = env_one_step(cur_env, action[:, i, j])
                if done:
                    break
                if (j + 1) % action_per_frame == 0:
                    full_obs_list.append(observes)
                    key_frame_list.append(observes)
            if done:
                break
        first = False
        if done:
            break
        if open_loop:
            model.infer(dict(compute_kv_cache=True, imagine=True))
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True,
                             imagine=False, state=action))

    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)
    save_video(full_obs_list, out_file, fps=60)
    with open(str(out_file).replace('.mp4', '_progress.json'), 'w') as f:
        json.dump(p_hat_traj, f)

    cur_env.close()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--libero-benchmark', type=str, default='libero_10')
    ap.add_argument('--task-idx', type=int, required=True,
                    help='single task (must match the demo pack)')
    ap.add_argument('--port', type=int, default=23908)
    ap.add_argument('--test-num', type=int, default=10)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--demo-states', type=str, required=True,
                    help='<pack>_states.npz sidecar from demo_encoder_icl.py')
    ap.add_argument('--search-ahead-steps', type=int, default=64,
                    help='GT matcher forward window (env steps)')
    ap.add_argument('--open-loop', action='store_true',
                    help='self-imagined KV advance (imagine=True), no real obs '
                         'feedback — isolates demo influence on video generation')
    args = ap.parse_args()

    sc = np.load(args.demo_states)
    matcher = GTProgressMatcher(sc['states'], sc['chunk_end_steps'],
                                search_ahead_steps=args.search_ahead_steps)

    model = WebsocketClientPolicy(port=args.port)
    succ_num = 0.
    for episode_idx in tqdm(range(args.test_num)):
        res = run_one(model, args.libero_benchmark, args.task_idx,
                      args.out_dir, episode_idx, matcher,
                      open_loop=args.open_loop)
        succ_num += res
        succ_rate = succ_num / (episode_idx + 1)
        print(f'Success rate: {succ_rate}, success num: {succ_num}, '
              f'total num: {episode_idx + 1}')
        out_file = Path(args.out_dir) / f'{args.libero_benchmark}_{args.task_idx}.json'
        out_file.parent.mkdir(exist_ok=True, parents=True)
        with open(out_file, 'w') as f:
            json.dump({'succ_num': succ_num, 'total_num': episode_idx + 1.,
                       'succ_rate': succ_rate}, f)


if __name__ == '__main__':
    main()
