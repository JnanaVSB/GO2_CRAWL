"""
Render a saved DMP weight vector in MuJoCo and save as MP4 + GIF.

Usage:
    python render_best.py --config configs/cmaes_bfs10.yaml
    python render_best.py --config configs/cmaes_bfs10.yaml --weights path/to/weights.npy
    python render_best.py --config configs/cmaes_bfs10.yaml --sim_steps 2000 --fps 30
"""

import yaml
import argparse
import os
import sys
import numpy as np
import mujoco
import mediapy
import imageio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.go2_env import Go2CrawlEnv
from policy.dmp_policy import DMPPolicy
from env.rewards import get_reward_fn, get_termination_fn


def render_rollout(env, policy, weights, sim_steps, reward_fn, termination_fn, reward_cfg, width=1280, height=720, camera_name=None):
    """
    Run a rollout and capture frames from MuJoCo's offscreen renderer.

    Returns
    -------
    frames : list of np.ndarray (H, W, 3)
    total_reward : float
    distance_x : float
    steps : int
    terminated : bool
    """
    joint_traj = policy.generate_trajectory(weights)
    traj_len = len(joint_traj)

# Override model framebuffer size to match requested resolution
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height

    # Set up offscreen renderer
    renderer = mujoco.Renderer(env.model, height=height, width=width)

    prev_obs, _ = env.reset()
    start_x = prev_obs[24]
    total_reward = 0.0
    terminated = False
    frames = []

    for t in range(sim_steps):
        target = joint_traj[t % traj_len]
        obs, _, _, _, _ = env.step(target)

        reward = reward_fn(prev_obs, obs, reward_cfg)
        total_reward += reward

        terminated = termination_fn(obs, reward_cfg)
        if terminated:
            total_reward -= reward_cfg.get("fall_penalty", 0.0)
            break

        prev_obs = obs

        # Capture frame
        if camera_name is not None:
            renderer.update_scene(env.data, camera=camera_name)
        else:
            renderer.update_scene(env.data)
        frame = renderer.render()
        frames.append(frame.copy())

    distance_x = obs[24] - start_x
    renderer.close()

    return frames, total_reward, distance_x, t + 1, terminated


def main():
    parser = argparse.ArgumentParser(description="Render best weights in MuJoCo")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to weights .npy file. Default: best_weights.npy in logdir",
    )
    parser.add_argument(
        "--sim_steps", type=int, default=None,
        help="Simulation steps. Default: from config",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second for video output",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Video width",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Video height",
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="MuJoCo camera name. Default: free camera",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory. Default: logdir/renders",
    )
    parser.add_argument(
        "--no_gif", action="store_true",
        help="Skip GIF generation (faster, smaller files)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    env_cfg = config["env"]
    policy_cfg = config["policy"]
    reward_cfg = config["reward"]
    train_cfg = config["training"]

    logdir = train_cfg["logdir"]
    sim_steps = args.sim_steps or train_cfg["sim_steps"]

    # Load weights
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = os.path.join(logdir, "best_weights.npy")

    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        sys.exit(1)

    weights = np.load(weights_path).flatten()
    print(f"Loaded weights: {weights_path} ({len(weights)} params)")

    # Output directory
    output_dir = args.output_dir or os.path.join(logdir, "renders")
    os.makedirs(output_dir, exist_ok=True)

    # Build env and policy
    env = Go2CrawlEnv(**env_cfg)
    policy = DMPPolicy(**policy_cfg)
    reward_fn = get_reward_fn(reward_cfg["reward_fn"])
    termination_fn = get_termination_fn(reward_cfg["termination_fn"])

    print(f"Running rollout: {sim_steps} steps...")

    frames, total_reward, distance_x, steps, terminated = render_rollout(
        env=env,
        policy=policy,
        weights=weights,
        sim_steps=sim_steps,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        reward_cfg=reward_cfg,
        width=args.width,
        height=args.height,
        camera_name=args.camera,
    )

    print(f"Rollout complete:")
    print(f"  Steps:      {steps}")
    print(f"  Reward:     {total_reward:.4f}")
    print(f"  Distance X: {distance_x:.4f}")
    print(f"  Terminated: {terminated}")
    print(f"  Frames:     {len(frames)}")

    if len(frames) == 0:
        print("No frames captured (robot may have terminated immediately).")
        sys.exit(1)

    # Subsample frames to match target fps
    # MuJoCo timestep * control_substeps gives sim time per frame
    dt = env.model.opt.timestep * env_cfg["control_substeps"]
    sim_fps = 1.0 / dt
    skip = max(1, int(sim_fps / args.fps))
    frames_out = frames[::skip]
    print(f"  Sim FPS:    {sim_fps:.0f}")
    print(f"  Output FPS: {args.fps} (every {skip} frames -> {len(frames_out)} frames)")

    # Save MP4
    mp4_path = os.path.join(output_dir, "rollout.mp4")
    imageio.mimsave(mp4_path, frames_out, fps=args.fps, codec="libx264")
    print(f"Saved: {mp4_path}")

    # Save GIF
    if not args.no_gif:
        gif_path = os.path.join(output_dir, "rollout.gif")
        # Downscale for GIF to keep file size reasonable
        gif_frames = []
        for frame in frames_out:
            h, w = frame.shape[:2]
            small = frame[::2, ::2]  # half resolution
            gif_frames.append(small)
        imageio.mimsave(gif_path, gif_frames, fps=args.fps, loop=0)
        print(f"Saved: {gif_path}")

    print("Done.")


if __name__ == "__main__":
    main()