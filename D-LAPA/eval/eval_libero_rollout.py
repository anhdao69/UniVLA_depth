"""
Closed-loop rollout evaluation of a fine-tuned LAPA model on LIBERO test tasks.

Architecture (two venvs, so model and simulator run in separate processes):
  * Model server: `python -m latent_pretraining.deploy ...` in the `lapa-depth` venv,
    serving POST /act  ({"image": <path>, "instruction": <str>} -> [7 float action]).
  * This client: runs in the `LIBERO` venv (mujoco/robosuite/libero). For each test
    task it creates the LIBERO sim, rolls out the policy by querying the server each
    step, records a video, and checks task success.

Run via scripts/eval_libero.sh (which starts the server and sets MUJOCO_GL/PYTHONPATH),
or standalone once a server is already up:

  MUJOCO_GL=egl PYTHONPATH=<LIBERO_repo> \
  python eval/eval_libero_rollout.py \
      --server_url http://127.0.0.1:32820/act \
      --output_dir outputs/eval_libero \
      --suites libero_spatial libero_object libero_goal libero_10

Outputs:
  {output_dir}/{suite}/{task}/ep{i}_{success|fail}.mp4   # one video per episode
  {output_dir}/results.json                              # per-task / per-suite / overall success
"""

import argparse
import json
import os
import tempfile
import time

import numpy as np
import requests
from PIL import Image

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - older imageio
    import imageio

# NOTE: libero / robosuite / mujoco are imported lazily inside main() so that
# --help and argument validation work without a GPU/EGL rendering context.


# Start index into each task's 50 fixed init states = the "test" split.
# spatial/object/goal: demos 45-49 are held out for test (matches data split).
# libero_10 / libero_90: the whole suite is a split, so start at 0.
TEST_INIT_OFFSET = {
    'libero_spatial': 45,
    'libero_object': 45,
    'libero_goal': 45,
    'libero_10': 0,
    'libero_90': 0,
}

# Per-suite rollout horizon (long-horizon suites need more steps).
DEFAULT_MAX_STEPS = {
    'libero_spatial': 520,
    'libero_object': 520,
    'libero_goal': 520,
    'libero_10': 620,
    'libero_90': 520,
}


def get_agentview_image(obs_dict):
    """Return the agentview RGB frame from a LIBERO obs dict (robust to key naming)."""
    if 'agentview_image' in obs_dict:
        return obs_dict['agentview_image']
    for k in obs_dict:
        if 'agentview' in k and 'image' in k:
            return obs_dict[k]
    for k in obs_dict:  # last resort: any rendered image
        if k.endswith('_image'):
            return obs_dict[k]
    raise KeyError(f'No agentview image in obs keys: {list(obs_dict.keys())}')


def query_action(server_url, image_model, instruction, tmp_path,
                 binarize_gripper, connect_retries, retry_wait):
    """Save the frame to a temp file (server reads it by path) and get an action back."""
    Image.fromarray(image_model.astype(np.uint8)).save(tmp_path)
    payload = {'image': tmp_path, 'instruction': instruction}
    last_err = None
    for _ in range(connect_retries):
        try:
            resp = requests.post(server_url, json=payload, timeout=180)
            data = resp.json()
            action = np.asarray(data, dtype=np.float32)
            if action.shape != (7,):
                raise ValueError(f'bad action from server: {data!r}')
            if binarize_gripper:
                action[6] = 1.0 if action[6] > 0 else -1.0
            return action
        except (requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            time.sleep(retry_wait)
    raise RuntimeError(f'server unreachable at {server_url}: {last_err}')


def make_env(bddl_folder, task, img_size):
    """Create a single LIBERO env wrapped in DummyVectorEnv (mirrors libero metric.py)."""
    env_args = {
        'bddl_file_name': os.path.join(bddl_folder, task.problem_folder, task.bddl_file),
        'camera_heights': img_size,
        'camera_widths': img_size,
    }
    env = None
    last_err = None
    for attempt in range(5):
        try:
            env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
            break
        except Exception as e:
            last_err = e
            print(f'[make_env] attempt {attempt} failed: {type(e).__name__}: {e}')
            time.sleep(5)
    if env is None:
        raise RuntimeError(
            f'failed to create env for {task.bddl_file}: {type(last_err).__name__}: {last_err}'
        ) from last_err
    return env


def rollout_episode(env, init_state, task, args, tmp_path):
    """Run one episode; return (success: bool, frames: list[np.ndarray])."""
    env.reset()
    obs = env.set_init_state(init_state[None])  # (1, D) -> obs is a list of 1 dict
    # Let physics settle with dummy zero actions (as in libero's own eval).
    for _ in range(5):
        obs, _, _, _ = env.step(np.zeros((1, 7)))

    frames = []
    success = False
    max_steps = args.max_steps if args.max_steps > 0 else DEFAULT_MAX_STEPS.get(args._suite, 520)
    for _ in range(max_steps):
        raw_img = np.asarray(get_agentview_image(obs[0]))  # opengl: stored upside-down
        # Feed the model the SAME orientation as the training images:
        #   rot180_for_model -> 180° rotation (OpenVLA-style img[::-1, ::-1])
        #   flip_for_model   -> vertical flip only
        #   neither          -> raw frame as returned by the env
        if args.rot180_for_model:
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
        frames.append(raw_img if args.rot180_for_model else raw_img[::-1, ::-1])

        action = query_action(
            args.server_url, image_model, task.language, tmp_path,
            args.binarize_gripper, args.connect_retries, args.retry_wait,
        )
        obs, _, done, _ = env.step(action[None])
        if bool(done[0]):
            success = True
            break
    return success, frames


def save_video(frames, path, fps):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps)
    except Exception:  # ffmpeg missing -> fall back to gif
        imageio.mimsave(os.path.splitext(path)[0] + '.gif', frames, fps=fps)


def sanitize(name):
    return ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)


def main():
    args = parse_args()
    # Deferred heavy imports (need a GPU/EGL context); keeps --help usable anywhere.
    global benchmark, get_libero_path, OffScreenRenderEnv, DummyVectorEnv
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv

    # torch>=2.6 flipped torch.load default to weights_only=True, which rejects the
    # numpy-pickled LIBERO init-state files. Restore the old behavior for LIBERO's loads.
    import torch
    _orig_torch_load = torch.load

    def _torch_load_compat(*a, **k):
        k.setdefault('weights_only', False)
        return _orig_torch_load(*a, **k)

    torch.load = _torch_load_compat

    os.makedirs(args.output_dir, exist_ok=True)
    bddl_folder = get_libero_path('bddl_files')
    benchmark_dict = benchmark.get_benchmark_dict()

    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    tmp_path = tmp.name
    tmp.close()

    results = {}
    total_success, total_episodes = 0, 0

    for suite in args.suites:
        if suite not in benchmark_dict:
            print(f'[warn] unknown suite {suite}, skipping')
            continue
        args._suite = suite
        b = benchmark_dict[suite]()
        offset = args.init_offset if args.init_offset >= 0 else TEST_INIT_OFFSET.get(suite, 0)
        suite_success, suite_episodes = 0, 0
        results[suite] = {'tasks': {}}

        for task_id in range(b.n_tasks):
            task = b.get_task(task_id)
            init_states = np.asarray(b.get_task_init_states(task_id))
            task_name = sanitize(task.bddl_file.replace('.bddl', ''))
            env = make_env(bddl_folder, task, args.img_size)

            task_success = 0
            for ep in range(args.n_eval_per_task):
                idx = (offset + ep) % init_states.shape[0]
                success, frames = rollout_episode(env, init_states[idx], task, args, tmp_path)
                task_success += int(success)
                tag = 'success' if success else 'fail'
                if success or args.save_failures:
                    save_video(
                        frames,
                        os.path.join(args.output_dir, suite, task_name, f'ep{ep}_{tag}.mp4'),
                        args.fps,
                    )
                print(f'[{suite}] task {task_id} ({task.language!r}) ep {ep}: {tag}')

            env.close()
            sr = task_success / args.n_eval_per_task
            results[suite]['tasks'][task_name] = {'success_rate': sr, 'n_eval': args.n_eval_per_task}
            suite_success += task_success
            suite_episodes += args.n_eval_per_task

        suite_sr = suite_success / max(suite_episodes, 1)
        results[suite]['success_rate'] = suite_sr
        results[suite]['n_eval'] = suite_episodes
        total_success += suite_success
        total_episodes += suite_episodes
        print(f'\n=== {suite}: success rate {suite_sr:.3f} ({suite_success}/{suite_episodes}) ===\n')

    results['overall'] = {
        'success_rate': total_success / max(total_episodes, 1),
        'n_eval': total_episodes,
    }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print('\n================ SUMMARY ================')
    for suite in args.suites:
        if suite in results:
            print(f'  {suite:16s}: {results[suite]["success_rate"]:.3f} '
                  f'({results[suite]["n_eval"]} eps)')
    print(f'  {"OVERALL":16s}: {results["overall"]["success_rate"]:.3f} '
          f'({total_episodes} eps)')
    print(f'Results + videos saved under {args.output_dir}')
    os.unlink(tmp_path)


def parse_args():
    p = argparse.ArgumentParser(description='LIBERO closed-loop rollout eval for LAPA.')
    p.add_argument('--server_url', type=str, default='http://127.0.0.1:32820/act',
                   help='LAPA deploy server /act endpoint.')
    p.add_argument('--output_dir', type=str, default='outputs/eval_libero')
    p.add_argument('--suites', type=str, nargs='+',
                   default=['libero_spatial', 'libero_object', 'libero_goal', 'libero_10'],
                   help='Benchmark suites to evaluate.')
    p.add_argument('--n_eval_per_task', type=int, default=5,
                   help='Rollouts per task (paper LIBERO-100 test uses ~50 for libero_10).')
    p.add_argument('--init_offset', type=int, default=-1,
                   help='Start index into the 50 init states. -1 = auto per split '
                        '(45 for spatial/object/goal, 0 for libero_10/90).')
    p.add_argument('--max_steps', type=int, default=-1,
                   help='Rollout horizon. -1 = per-suite default.')
    p.add_argument('--img_size', type=int, default=128,
                   help='Camera render size (matches training agentview_rgb = 128).')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--save_failures', action='store_true', default=True,
                   help='Also save videos for failed episodes.')
    p.add_argument('--no_save_failures', dest='save_failures', action='store_false')
    p.add_argument('--flip_for_model', action='store_true', default=False,
                   help='Vertically flip frames before sending to the model. Default False '
                        '(training used raw upside-down agentview_rgb).')
    p.add_argument('--rot180_for_model', action='store_true', default=False,
                   help='Rotate frames 180 degrees (img[::-1, ::-1], OpenVLA convention) '
                        'before sending to the model. Takes precedence over --flip_for_model.')
    p.add_argument('--binarize_gripper', action='store_true', default=True,
                   help='Snap gripper action to +/-1 by sign (matches LIBERO +/-1 convention).')
    p.add_argument('--no_binarize_gripper', dest='binarize_gripper', action='store_false')
    p.add_argument('--connect_retries', type=int, default=60,
                   help='Retries per request while waiting for the server to come up.')
    p.add_argument('--retry_wait', type=float, default=10.0)
    return p.parse_args()


if __name__ == '__main__':
    main()
