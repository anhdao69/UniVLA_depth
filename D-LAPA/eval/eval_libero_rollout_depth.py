"""
Depth-aware LIBERO rollout client for LAPA-Depth.

This mirrors ZaMin/LAPA/eval/eval_libero_rollout.py, but targets a policy server
that computes online depth features through DepthAnythingV2 + Stage-2.5.
"""

import argparse
import json
import os
import time
import tempfile

import numpy as np
import requests
from PIL import Image

try:
    import imageio.v2 as imageio
except Exception:
    import imageio


TEST_INIT_OFFSET = {
    "libero_spatial": 45,
    "libero_object": 45,
    "libero_goal": 45,
    "libero_10": 0,
    "libero_90": 0,
}

DEFAULT_MAX_STEPS = {
    "libero_spatial": 520,
    "libero_object": 520,
    "libero_goal": 520,
    "libero_10": 620,
    "libero_90": 520,
}


def get_agentview_image(obs_dict):
    if "agentview_image" in obs_dict:
        return obs_dict["agentview_image"]
    for key in obs_dict:
        if "agentview" in key and "image" in key:
            return obs_dict[key]
    for key in obs_dict:
        if key.endswith("_image"):
            return obs_dict[key]
    raise KeyError(f"No agentview image in obs keys: {list(obs_dict.keys())}")


def query_action(args, image_model, instruction, tmp_path):
    Image.fromarray(image_model.astype(np.uint8)).save(tmp_path)
    payload = {"image": tmp_path, "instruction": instruction}
    last_err = None
    for _ in range(args.connect_retries):
        try:
            resp = requests.post(args.server_url, json=payload, timeout=180)
            data = resp.json()
            if isinstance(data, dict) and "error" in data:
                raise RuntimeError(data["error"])
            if isinstance(data, str):
                raise RuntimeError(f"policy server returned string response: {data!r}")
            action = np.asarray(data, dtype=np.float32)
            if action.shape != (7,):
                raise ValueError(f"bad action from server: {data!r}")
            if args.binarize_gripper:
                action[6] = 1.0 if action[6] > 0 else -1.0
            return action
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_err = exc
            time.sleep(args.retry_wait)
    raise RuntimeError(f"server unreachable at {args.server_url}: {last_err}")


def make_env(bddl_folder, task, img_size):
    env_args = {
        "bddl_file_name": os.path.join(bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": img_size,
        "camera_widths": img_size,
    }
    env = None
    last_err = None
    for attempt in range(5):
        try:
            env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
            break
        except Exception as exc:
            last_err = exc
            print(f"[make_env] attempt {attempt} failed: {type(exc).__name__}: {exc}")
            time.sleep(5)
    if env is None:
        raise RuntimeError(f"failed to create env for {task.bddl_file}: {last_err}") from last_err
    return env


def rollout_episode(env, init_state, task, args, tmp_path, task_id, ep):
    env.reset()
    obs = env.set_init_state(init_state[None])
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros((1, 7)))

    frames = []
    success = False
    max_steps = args.max_steps if args.max_steps > 0 else DEFAULT_MAX_STEPS.get(args._suite, 520)
    episode_start = time.time()
    print(
        f"[{args._suite}] task {task_id} ep {ep} start | max_steps={max_steps} | "
        f"instruction={task.language!r}",
        flush=True,
    )
    for step_index in range(max_steps):
        raw_img = np.asarray(get_agentview_image(obs[0]))
        # Feed the model the SAME orientation as the training images:
        #   rot180_for_model -> 180° rotation (OpenVLA-style img[::-1, ::-1])
        #   flip_for_model   -> vertical flip only
        #   neither          -> raw frame as returned by the env
        if getattr(args, 'rot180_for_model', False):
            image_model = raw_img[::-1, ::-1]
        elif args.flip_for_model:
            image_model = raw_img[::-1]
        else:
            image_model = raw_img
        # Video orientation is tied to rot180_for_model so the saved video is
        # always human-upright:
        #   rot180_for_model=1 -> model already got the 180° rotation, so save the
        #                         raw env frame as-is (no postprocess).
        #   rot180_for_model=0 -> model got the raw frame, so rotate the video 180°.
        frames.append(raw_img if getattr(args, 'rot180_for_model', False) else raw_img[::-1, ::-1])
        action = query_action(args, image_model, task.language, tmp_path)
        obs, _, done, _ = env.step(action[None])
        if args.progress_freq > 0 and (
            (step_index + 1) % args.progress_freq == 0 or step_index == 0
        ):
            elapsed = time.time() - episode_start
            steps_done = step_index + 1
            sec_per_step = elapsed / max(steps_done, 1)
            eta = sec_per_step * max(max_steps - steps_done, 0)
            print(
                f"[{args._suite}] task {task_id} ep {ep} step {steps_done}/{max_steps} | "
                f"elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m | "
                f"sec_step={sec_per_step:.2f}",
                flush=True,
            )
        if bool(done[0]):
            success = True
            break
    return success, frames


def save_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception:
        imageio.mimsave(os.path.splitext(path)[0] + ".gif", frames, fps=fps)


def sanitize(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def main():
    args = parse_args()
    global benchmark, get_libero_path, OffScreenRenderEnv, DummyVectorEnv
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv

    import torch

    original_torch_load = torch.load

    def torch_load_compat(*call_args, **call_kwargs):
        call_kwargs.setdefault("weights_only", False)
        return original_torch_load(*call_args, **call_kwargs)

    torch.load = torch_load_compat

    os.makedirs(args.output_dir, exist_ok=True)
    bddl_folder = get_libero_path("bddl_files")
    benchmark_dict = benchmark.get_benchmark_dict()
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_path = tmp.name
    tmp.close()

    results = {}
    total_success = 0
    total_episodes = 0
    for suite in args.suites:
        if suite not in benchmark_dict:
            print(f"[warn] unknown suite {suite}, skipping")
            continue
        args._suite = suite
        bench = benchmark_dict[suite]()
        offset = args.init_offset if args.init_offset >= 0 else TEST_INIT_OFFSET.get(suite, 0)
        task_ids = args.task_ids if args.task_ids else list(range(bench.n_tasks))
        if args.max_tasks > 0:
            task_ids = task_ids[: args.max_tasks]
        suite_success = 0
        suite_episodes = 0
        results[suite] = {"tasks": {}}
        for task_id in task_ids:
            task = bench.get_task(task_id)
            init_states = np.asarray(bench.get_task_init_states(task_id))
            task_name = sanitize(task.bddl_file.replace(".bddl", ""))
            env = make_env(bddl_folder, task, args.img_size)
            task_success = 0
            for ep in range(args.n_eval_per_task):
                idx = (offset + ep) % init_states.shape[0]
                ep_start = time.time()
                success, frames = rollout_episode(env, init_states[idx], task, args, tmp_path, task_id, ep)
                task_success += int(success)
                tag = "success" if success else "fail"
                if success or args.save_failures:
                    save_video(
                        frames,
                        os.path.join(args.output_dir, suite, task_name, f"ep{ep}_{tag}.mp4"),
                        args.fps,
                    )
                print(
                    f"[{suite}] task {task_id} ({task.language!r}) ep {ep}: {tag} | "
                    f"episode_time={(time.time() - ep_start)/60:.1f}m",
                    flush=True,
                )
            env.close()
            rate = task_success / args.n_eval_per_task
            results[suite]["tasks"][task_name] = {"success_rate": rate, "n_eval": args.n_eval_per_task}
            suite_success += task_success
            suite_episodes += args.n_eval_per_task
        results[suite]["success_rate"] = suite_success / max(suite_episodes, 1)
        results[suite]["n_eval"] = suite_episodes
        total_success += suite_success
        total_episodes += suite_episodes

    results["overall"] = {"success_rate": total_success / max(total_episodes, 1), "n_eval": total_episodes}
    with open(os.path.join(args.output_dir, "results.json"), "w") as fout:
        json.dump(results, fout, indent=2)
    print(json.dumps(results["overall"], indent=2))
    os.unlink(tmp_path)


def parse_args():
    parser = argparse.ArgumentParser(description="LIBERO rollout eval for LAPA-Depth.")
    parser.add_argument("--server_url", type=str, default="http://127.0.0.1:32820/act")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_libero_depth")
    parser.add_argument("--suites", type=str, nargs="+", default=["libero_90"])
    parser.add_argument("--task_ids", type=int, nargs="*", default=[])
    parser.add_argument("--max_tasks", type=int, default=-1)
    parser.add_argument("--n_eval_per_task", type=int, default=1)
    parser.add_argument("--init_offset", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--save_failures", action="store_true", default=True)
    parser.add_argument("--no_save_failures", dest="save_failures", action="store_false")
    parser.add_argument("--flip_for_model", action="store_true", default=False)
    parser.add_argument("--rot180_for_model", action="store_true", default=False,
                        help="Rotate frames 180 degrees (img[::-1, ::-1], OpenVLA convention) before sending to the model.")
    parser.add_argument("--binarize_gripper", action="store_true", default=True)
    parser.add_argument("--no_binarize_gripper", dest="binarize_gripper", action="store_false")
    parser.add_argument("--connect_retries", type=int, default=60)
    parser.add_argument("--retry_wait", type=float, default=10.0)
    parser.add_argument("--progress_freq", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    main()
