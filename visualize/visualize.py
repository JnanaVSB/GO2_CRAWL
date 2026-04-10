"""
Visualization script for Go2 Crawl training results.

Generates plots from saved checkpoints and logs:
    1. Reward curve (reward vs generation)
    2. Distance curve (distance_x vs generation)
    3. PCA trajectory comparison (initial vs best)
    4. Joint angle trajectories (initial vs best)
    5. Weight evolution across checkpoints

Usage:
    python visualize.py --config configs/cmaes_bfs10.yaml
"""

import yaml
import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policy.dmp_policy import DMPPolicy


def _load_histories(logdir):
    """
    Load reward_history and dist_history from whichever format exists.

    Returns (reward_history, dist_history) as numpy arrays, or (None, None).

    SAC/PPO/ProPS runners save histories in checkpoints/histories.npz.
    Evolutionary runner stores them inside checkpoint_latest.npz.
    """
    checkpoint_dir = os.path.join(logdir, "checkpoints")

    # Try histories.npz first (SAC, PPO, ProPS format)
    hist_path = os.path.join(checkpoint_dir, "histories.npz")
    if os.path.exists(hist_path):
        hist = np.load(hist_path, allow_pickle=True)
        reward_history = hist.get("reward_history", None)
        dist_history = hist.get("dist_history", None)
        if reward_history is not None:
            return reward_history, dist_history
    
    # Fall back to checkpoint_latest.npz (evolutionary format)
    latest = os.path.join(checkpoint_dir, "checkpoint_latest.npz")
    if os.path.exists(latest):
        ckpt = np.load(latest, allow_pickle=True)
        reward_history = ckpt.get("reward_history", None)
        dist_history = ckpt.get("dist_history", None)
        if reward_history is not None:
            return reward_history, dist_history

    return None, None


def load_checkpoints(checkpoint_dir):
    """
    Load all numbered checkpoints sorted by step number.
    Works with all runner formats:
        - evolutionary: checkpoint_gen_NNNN.npz
        - SAC/PPO:      checkpoint_ep_NNNNNN_meta.npz
        - ProPS:        checkpoint_iter_NNNNNN_meta.npz
    """
    checkpoints = []

    # Evolutionary: checkpoint_gen_NNNN.npz
    for f in sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_gen_*.npz"))):
        ckpt = np.load(f, allow_pickle=True)
        checkpoints.append({
            "step": int(ckpt["gen"]),
            "best_reward": float(ckpt["best_reward"]),
            "best_weights": ckpt["best_weights"],
        })

    # SAC/PPO: checkpoint_ep_NNNNNN_meta.npz
    for f in sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_ep_*_meta.npz"))):
        meta = np.load(f, allow_pickle=True)
        bw = meta["best_weights"]
        checkpoints.append({
            "step": int(meta["episodes_done"]),
            "best_reward": float(meta["best_reward"]),
            "best_weights": bw if bw.size > 0 else None,
        })

    # ProPS: checkpoint_iter_NNNNNN_meta.npz
    for f in sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_iter_*_meta.npz"))):
        meta = np.load(f, allow_pickle=True)
        bw = meta["best_weights"]
        checkpoints.append({
            "step": int(meta["iterations_done"]),
            "best_reward": float(meta["best_reward"]),
            "best_weights": bw if bw.size > 0 else None,
        })

    # Sort by step number
    checkpoints.sort(key=lambda c: c["step"])
    return checkpoints


def plot_reward_curve(logdir, save_dir):
    """Plot reward vs iteration from history data."""
    reward_history, _ = _load_histories(logdir)
    if reward_history is None:
        print("No reward history found, skipping reward curve.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, "b-", alpha=0.3, linewidth=1, label="Reward per step")

    window = min(20, max(1, len(reward_history) // 5))
    if window > 1 and len(reward_history) > window:
        avg = np.convolve(reward_history, np.ones(window) / window, mode="valid")
        plt.plot(
            np.arange(window - 1, len(reward_history)),
            avg, "b-", linewidth=2, label=f"Running avg ({window})",
        )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_curve.png"), dpi=150)
    plt.close()
    print("Saved: reward_curve.png")


def plot_distance_curve(logdir, save_dir):
    """Plot distance_x vs iteration from history data."""
    _, dist_history = _load_histories(logdir)
    if dist_history is None:
        print("No distance history found, skipping distance curve.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(dist_history, "g-", alpha=0.3, linewidth=1, label="Distance per step")

    window = min(20, max(1, len(dist_history) // 5))
    if window > 1 and len(dist_history) > window:
        avg = np.convolve(dist_history, np.ones(window) / window, mode="valid")
        plt.plot(
            np.arange(window - 1, len(dist_history)),
            avg, "g-", linewidth=2, label=f"Running avg ({window})",
        )

    plt.xlabel("Step")
    plt.ylabel("Distance X (m)")
    plt.title("Forward Distance Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "distance_curve.png"), dpi=150)
    plt.close()
    print("Saved: distance_curve.png")


def plot_pca_comparison(policy_kwargs, initial_weights_path, logdir, save_dir):
    """Plot initial vs best trajectory in PCA space."""
    best_weights_path = os.path.join(logdir, "best_weights.npy")
    if not os.path.exists(best_weights_path):
        print("No best weights found, skipping PCA comparison.")
        return

    policy = DMPPolicy(**policy_kwargs)
    initial_weights = np.load(initial_weights_path).flatten()
    best_weights = np.load(best_weights_path).flatten()

    latent_initial, _ = policy.generate_latent_trajectory(initial_weights)
    latent_best, _ = policy.generate_latent_trajectory(best_weights)

    plt.figure(figsize=(10, 10))

    # Dataset poses
    plt.scatter(
        policy.X_pca[:, 0], policy.X_pca[:, 1],
        s=60, c="blue", zorder=5, label="Dataset poses",
    )
    for i, lbl in enumerate(policy.labels):
        plt.text(
            policy.X_pca[i, 0] + 0.02, policy.X_pca[i, 1] + 0.02,
            lbl, fontsize=7,
        )

    # Initial trajectory
    plt.plot(
        latent_initial[:, 0], latent_initial[:, 1],
        "g--", linewidth=2, alpha=0.5, label="Initial trajectory",
    )

    # Best trajectory
    plt.plot(
        latent_best[:, 0], latent_best[:, 1],
        "r-", linewidth=2, label="Best trajectory",
    )
    plt.scatter(
        latent_best[0, 0], latent_best[0, 1],
        s=100, c="orange", zorder=6, label="Start",
    )

    plt.xlabel(f"PC1 ({100 * policy.pca.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100 * policy.pca.explained_variance_ratio_[1]:.1f}%)")
    plt.title("PCA Space: Initial vs Best Trajectory")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pca_comparison.png"), dpi=150)
    plt.close()
    print("Saved: pca_comparison.png")


def plot_joint_trajectories(policy_kwargs, initial_weights_path, logdir, save_dir):
    """Plot 12 joint angle trajectories for initial vs best."""
    best_weights_path = os.path.join(logdir, "best_weights.npy")
    if not os.path.exists(best_weights_path):
        print("No best weights found, skipping joint trajectories.")
        return

    policy = DMPPolicy(**policy_kwargs)
    initial_weights = np.load(initial_weights_path).flatten()
    best_weights = np.load(best_weights_path).flatten()

    joint_initial = policy.generate_trajectory(initial_weights)
    joint_best = policy.generate_trajectory(best_weights)

    joint_names = [
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    ]

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for i, ax in enumerate(axes.flat):
        ax.plot(joint_initial[:, i], "g--", alpha=0.5, label="Initial")
        ax.plot(joint_best[:, i], "r-", label="Best")
        ax.set_title(joint_names[i])
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Angle (rad)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle("Joint Angle Trajectories: Initial vs Best", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "joint_trajectories.png"), dpi=150)
    plt.close()
    print("Saved: joint_trajectories.png")


def plot_weight_evolution(policy_kwargs, logdir, save_dir):
    """Plot how DMP weights change across checkpoints."""
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    checkpoints = load_checkpoints(checkpoint_dir)

    # Filter out checkpoints with no best_weights
    checkpoints = [c for c in checkpoints if c["best_weights"] is not None]

    if len(checkpoints) < 2:
        print("Not enough checkpoints for weight evolution plot.")
        return

    steps = [c["step"] for c in checkpoints]
    weights_matrix = np.array([c["best_weights"].flatten() for c in checkpoints])
    n_params = weights_matrix.shape[1]

    plt.figure(figsize=(12, 6))
    for i in range(n_params):
        plt.plot(steps, weights_matrix[:, i], "-o", markersize=3, label=f"w[{i}]")

    plt.xlabel("Step")
    plt.ylabel("Weight Value")
    plt.title("DMP Weight Evolution Across Checkpoints")
    plt.grid(True, alpha=0.3)
    if n_params <= 20:
        plt.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_evolution.png"), dpi=150)
    plt.close()
    print("Saved: weight_evolution.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    policy_kwargs = config["policy"]
    agent_cfg = config["agent"]
    logdir = config["training"]["logdir"]
    initial_weights_path = agent_cfg["initial_weights_path"]

    save_dir = os.path.join(logdir, "visualizations")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Generating visualizations from: {logdir}")
    print(f"Saving to: {save_dir}")
    print()

    plot_reward_curve(logdir, save_dir)
    plot_distance_curve(logdir, save_dir)
    plot_pca_comparison(policy_kwargs, initial_weights_path, logdir, save_dir)
    plot_joint_trajectories(policy_kwargs, initial_weights_path, logdir, save_dir)
    plot_weight_evolution(policy_kwargs, logdir, save_dir)

    print(f"\nAll visualizations saved to: {save_dir}")


if __name__ == "__main__":
    main()