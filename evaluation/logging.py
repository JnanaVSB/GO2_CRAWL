"""
Logging and plotting utilities for Go2 Crawl training.

Shared by all runners so that log format, checkpointing, and plots
are consistent regardless of which optimizer is used.
"""

import os
import datetime
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_log_header(log_path, config, columns):
    """
    Write config header and CSV column line to a log file.

    Parameters
    ----------
    log_path : str
        Path to the CSV log file (appended to).
    config : dict
        Full config dict to dump in the header.
    columns : str
        CSV column header line (e.g. "timestamp,episode,reward,...").
        Each runner passes its own column string since they track
        different metrics.
    """
    with open(log_path, "a") as f:
        f.write("\n")
        f.write("#" * 60 + "\n")
        f.write(f"# Run started: {datetime.datetime.now().isoformat()}\n")
        f.write("#" * 60 + "\n")
        f.write(yaml.dump(config, default_flow_style=False))
        f.write("#" * 60 + "\n")
        f.write(columns + "\n")


def save_pca_plot(policy_cls, policy_kwargs, weights, iteration_label,
                  reward, dist_x, save_path):
    """
    Save PCA trajectory plot for current best weights.

    Only works when the policy has PCA attributes (X_pca, labels, pca).
    Silently skips if the policy does not support latent trajectory
    visualization (e.g. direct DMP mode).

    Parameters
    ----------
    policy_cls : class
        Trajectory generator / policy class.
    policy_kwargs : dict
        Constructor kwargs.
    weights : np.ndarray
        Flat weight vector to visualize.
    iteration_label : str
        Label for the plot title (e.g. "Gen 50", "Ep 100", "Iter 20").
    reward : float
        Best reward value for the title.
    dist_x : float
        Distance in x for the title.
    save_path : str
        File path to save the plot.
    """
    policy = policy_cls(**policy_kwargs)

    # Check if this policy supports latent trajectory visualization
    if not hasattr(policy, "generate_latent_trajectory"):
        return
    if not hasattr(policy, "X_pca"):
        return

    latent_traj, _ = policy.generate_latent_trajectory(weights)

    plt.figure(figsize=(8, 8))

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

    # DMP trajectory
    plt.plot(
        latent_traj[:, 0], latent_traj[:, 1],
        "r-", linewidth=2, label="DMP trajectory",
    )
    plt.scatter(
        latent_traj[0, 0], latent_traj[0, 1],
        s=100, c="orange", zorder=6, label="Start",
    )

    plt.xlabel(f"PC1 ({100 * policy.pca.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100 * policy.pca.explained_variance_ratio_[1]:.1f}%)")
    plt.title(f"{iteration_label} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def get_pca_range(policy_cls, policy_kwargs, weights):
    """
    Compute PCA latent trajectory range for a given weight vector.
    Used for the 'detailed' feedback mode in ProPS.

    Returns
    -------
    pca_x_range : tuple (min, max) or None
    pca_y_range : tuple (min, max) or None

    Returns (None, None) if the policy does not support latent trajectories.
    """
    policy = policy_cls(**policy_kwargs)

    if not hasattr(policy, "generate_latent_trajectory"):
        return None, None

    latent_traj, _ = policy.generate_latent_trajectory(weights)
    pca_x_range = (float(latent_traj[:, 0].min()), float(latent_traj[:, 0].max()))
    pca_y_range = (float(latent_traj[:, 1].min()), float(latent_traj[:, 1].max()))
    return pca_x_range, pca_y_range


def setup_logdir(logdir):
    """
    Create the standard directory structure for a training run.

    Returns
    -------
    checkpoint_dir : str
    plots_dir : str
    """
    os.makedirs(logdir, exist_ok=True)
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(logdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    return checkpoint_dir, plots_dir