"""
LIBERO eval client for OOD (out-of-distribution) text-prompt behavior testing.

Uses libero_goal task 8 ("put the bowl on the plate") as the fixed env/init-state
source -- its bddl scene contains cream_cheese, a wooden_cabinet (top/middle/bottom
drawers), a stove, and a wine rack/bottle, i.e. every object referenced by the OOD
prompts below -- but overrides the prompt sent to the model with an OOD variant
instead of task 8's real language ("put the bowl on the plate"). `done` is still
task 8's own env goal check (bowl on plate), so a True flag under an OOD prompt
means the model ignored the prompt and fell back to its trained default behavior.

Everything else (env driving, KV-cache feedback, action clipping, video saving)
matches evaluation/libero/client.py closed-loop behavior exactly; output layout
also matches it (benchmark/{idx}_{prompt}/{episode}_{done}.mp4) so
scripts/pair_eval_videos.py can pair real/predicted videos unmodified.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from libero.libero import benchmark
from lerobot.datasets.utils import write_json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from client import construct_single_env, init_single_env, env_one_step, save_video
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy

# (out_idx, prompt) -- out_idx is only used to prefix the output directory name,
# consistently with client.py's f"{task_idx}_{prompt}" scheme.
OOD_PROMPTS = [
    (0, "put the cream cheese in the plate"),          # variant 1: new goal, same scene
    (1, "open the top drawer and put the cream cheese inside"),  # variant 2: new goal + new action sequence
    (2, "pick up the cream cheese and put it on the stove"),      # variant 3: cross-object mis-binding
    (3, "open the top drawer"),                        # partial prompt: first half of variant 2 only
    (4, "wash the car"),                                # random prompt: unrelated to the scene
    (5, ""),                                            # empty prompt
]


def run_one(model, libero_benchmark, env_task_idx, prompt, out_idx, out_dir, episode_idx):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(env_task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    init_states = benchmark_instance.get_task_init_states(env_task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        if first:
            ret = model.infer(dict(obs=first_obs, prompt=prompt))
        else:
            ret = model.infer(dict(prompt=prompt))
        # Clip to the env action range BEFORE execution so the actions fed back
        # into the KV cache (state=action below) are exactly what the env ran.
        action = np.clip(ret['action'], -1.0, 1.0)

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
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

        model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    prompt_dir = prompt.replace(' ', '_')
    out_file = Path(out_dir) / libero_benchmark / f"{out_idx}_{prompt_dir}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_file,
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )

    cur_env.close()
    return done


def run(libero_benchmark, env_task_idx, port, out_dir, test_num):
    model = WebsocketClientPolicy(port=port)

    print(f"#################### OOD prompt eval on {libero_benchmark} task {env_task_idx}'s "
          f"env/init-state, {len(OOD_PROMPTS)} prompt variants x {test_num} episodes #############")

    for out_idx, prompt in tqdm(OOD_PROMPTS, total=len(OOD_PROMPTS)):
        succ_num = 0.
        for episode_idx in tqdm(range(test_num), total=test_num, leave=False):
            res_i = run_one(model, libero_benchmark, env_task_idx, prompt, out_idx, out_dir, episode_idx)
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"[{prompt!r}] Success rate (vs task {env_task_idx}'s own goal): {succ_rate}, "
                  f"success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{out_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "prompt": prompt,
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-benchmark", type=str, default="libero_goal")
    parser.add_argument("--env-task-idx", type=int, default=8,
                         help="Task index whose bddl scene/init-states are used as the fixed env "
                              "(default 8 = 'put the bowl on the plate', whose scene contains all "
                              "objects the OOD prompts reference).")
    parser.add_argument("--port", type=int, default=23908)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="outputs/ood_prompts")
    args = parser.parse_args()
    run(args.libero_benchmark, args.env_task_idx, args.port, args.out_dir, args.test_num)
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
