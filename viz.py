"""
Go2 Crawl — Visualization Entry Point.

Single command-line tool for generating all visualizations.

Usage:
    python viz.py --config configs/cmaes_bfs10.yaml --graphs
    python viz.py --config configs/cmaes_bfs10.yaml --render
    python viz.py --config configs/cmaes_bfs10.yaml --pca_gif
    python viz.py --config configs/cmaes_bfs10.yaml --graphs --render

At least one of --graphs, --render, --pca_gif is required.
"""

import yaml
import argparse
import os
import sys

from trajectory.pca_dmp import PCADMPTrajectory
from trajectory.direct_dmp import DirectDMPTrajectory
from visualize.graphs import generate_all_graphs
from visualize.render import run_render
from visualize.pca_gif import generate_pca_gif


# Same registry as main.py
TRAJECTORY_REGISTRY = {
    "pca_dmp": PCADMPTrajectory,
    "direct_dmp": DirectDMPTrajectory,
}


def get_trajectory_cls(config):
    """Look up trajectory generator class from config."""
    traj_cfg = config.get("trajectory", {})
    traj_type = traj_cfg.get("type", "pca_dmp")

    if traj_type not in TRAJECTORY_REGISTRY:
        available = ", ".join(TRAJECTORY_REGISTRY.keys())
        raise ValueError(
            f"Unknown trajectory type: '{traj_type}'. Available: {available}"
        )

    return TRAJECTORY_REGISTRY[traj_type]


def main():
    parser = argparse.ArgumentParser(
        description="Go2 Crawl visualization tool",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the config YAML file",
    )

    # Action flags
    parser.add_argument(
        "--graphs", action="store_true",
        help="Generate all training graphs (reward, distance, PCA, joints, weights)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Render MuJoCo rollout to MP4/GIF",
    )
    parser.add_argument(
        "--pca_gif", action="store_true",
        help="Animate PCA checkpoint plots into a GIF",
    )

    # Render options
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to weights .npy file (for --render). Default: best_weights.npy",
    )
    parser.add_argument(
        "--sim_steps", type=int, default=None,
        help="Simulation steps (for --render). Default: from config",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frames per second (for --render and --pca_gif). Default: 30",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Video width (for --render). Default: 1280",
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Video height (for --render). Default: 720",
    )
    parser.add_argument(
        "--camera", type=str, default=None,
        help="MuJoCo camera name (for --render). Default: free camera",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (for --render). Default: logdir/renders",
    )
    parser.add_argument(
        "--no_gif", action="store_true",
        help="Skip GIF generation in --render (faster, smaller files)",
    )

    args = parser.parse_args()

    # Require at least one action
    if not (args.graphs or args.render or args.pca_gif):
        parser.error("At least one of --graphs, --render, --pca_gif is required.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trajectory_cls = get_trajectory_cls(config)
    logdir = config["training"]["logdir"]

    # --graphs
    if args.graphs:
        policy_kwargs = config["policy"]
        initial_weights_path = config["agent"]["initial_weights_path"]
        save_dir = os.path.join(logdir, "visualizations")

        generate_all_graphs(
            trajectory_cls=trajectory_cls,
            policy_kwargs=policy_kwargs,
            initial_weights_path=initial_weights_path,
            logdir=logdir,
            save_dir=save_dir,
        )
        print()

    # --render
    if args.render:
        run_render(
            trajectory_cls=trajectory_cls,
            config=config,
            weights_path=args.weights,
            sim_steps=args.sim_steps,
            fps=args.fps,
            width=args.width,
            height=args.height,
            camera=args.camera,
            output_dir=args.output_dir,
            no_gif=args.no_gif,
        )
        print()

    # --pca_gif
    if args.pca_gif:
        pca_fps = min(args.fps, 3)  # PCA gif looks better slow
        generate_pca_gif(
            logdir=logdir,
            fps=pca_fps,
        )
        print()


if __name__ == "__main__":
    main()