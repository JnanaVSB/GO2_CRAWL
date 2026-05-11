"""
Per-iteration and post-training artifact generation.

Two entry points:

- save_iteration_artifacts(...)
    Called from inside a training loop. For one weight vector, writes:
        <iter_dir>/video.mp4           short MuJoCo rollout video
        <iter_dir>/weights_heatmap.png reshape(weights, (n_components, n_bfs))
        <iter_dir>/latent.png          dataset poses + rollout in PCA space
        <iter_dir>/info.txt            iteration number, reward, distance
    All four are written under <logdir>/iterations/iter_<NNNNNN>/.

- save_reward_curve(log_path, out_path, runner_type)
    Called once at end of training. Reads training_log.csv, plots the
    reward curve, marks the best iteration with a red star.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# Per-iteration helpers
# -----------------------------------------------------------------------

def _save_video(weights, iter_dir, env_cls, env_kwargs, policy_cls, policy_kwargs,
                reward_fn_name, termination_fn_name, reward_cfg, sim_steps,
                fps=30, width=640, height=360):
    """
    Record a short MuJoCo rollout video. Uses visualize.render.render_rollout
    so we get the same camera/rendering behavior as the standalone tool.
    """
    try:
        import imageio
        from visualize.render import render_rollout
        from env.rewards import get_reward_fn, get_termination_fn
    except Exception as e:
        print(f"  [video] skipped: {e}")
        return

    reward_fn = get_reward_fn(reward_fn_name)
    termination_fn = get_termination_fn(termination_fn_name)

    env = env_cls(**env_kwargs)
    policy = policy_cls(**policy_kwargs)

    try:
        frames, _, _, _, _ = render_rollout(
            env, policy, weights, sim_steps,
            reward_fn, termination_fn, reward_cfg,
            width=width, height=height,
        )
    except Exception as e:
        print(f"  [video] render failed: {e}")
        return

    if not frames:
        return

    out = os.path.join(iter_dir, "video.mp4")
    try:
        imageio.mimsave(out, frames, fps=fps, codec="libx264", quality=7)
    except Exception:
        # Fallback if libx264 not available: write a GIF instead
        out_gif = os.path.join(iter_dir, "video.gif")
        imageio.mimsave(out_gif, frames, fps=fps)


def _save_weights_heatmap(weights, iter_dir, n_components, n_bfs):
    """
    Save weights reshaped to (n_components, n_bfs) as a heatmap.
    """
    W = np.asarray(weights, dtype=np.float64).reshape(n_components, n_bfs)
    fig, ax = plt.subplots(figsize=(max(6, n_bfs * 0.5), max(2, n_components * 0.6)))
    vmax = np.max(np.abs(W)) if np.max(np.abs(W)) > 0 else 1.0
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("basis function")
    ax.set_ylabel("PCA component")
    ax.set_xticks(np.arange(n_bfs))
    ax.set_yticks(np.arange(n_components))
    for i in range(n_components):
        for j in range(n_bfs):
            ax.text(j, i, f"{W[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    plt.colorbar(im, ax=ax)
    ax.set_title("DMP weights")
    plt.tight_layout()
    plt.savefig(os.path.join(iter_dir, "weights_heatmap.png"), dpi=110)
    plt.close()


def _save_latent_plot(weights, iter_dir, policy_cls, policy_kwargs):
    """
    Plot dataset poses + DMP rollout in PCA latent space.
    Goal = pca_mean (red X). Start = the point on the rollout
    trajectory nearest to X_pca[0] (orange star).
    """
    policy = policy_cls(**policy_kwargs)
    if not hasattr(policy, "generate_latent_trajectory"):
        return
    if not hasattr(policy, "X_pca"):
        return

    latent_traj, _ = policy.generate_latent_trajectory(weights)

    pca_mean = np.mean(policy.X_pca, axis=0)
    first_pose = policy.X_pca[0]
    # nearest point on the rollout to X_pca[0]
    dists = np.linalg.norm(latent_traj - first_pose, axis=1)
    start_idx = int(np.argmin(dists))
    start_pt = latent_traj[start_idx]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(policy.X_pca[:, 0], policy.X_pca[:, 1],
               s=50, c="steelblue", zorder=4, label="Dataset poses")
    for i, lbl in enumerate(policy.labels):
        ax.annotate(lbl, (policy.X_pca[i, 0], policy.X_pca[i, 1]),
                    xytext=(4, 4), textcoords="offset points", fontsize=6)
    ax.plot(latent_traj[:, 0], latent_traj[:, 1],
            "r-", lw=1.8, label="DMP rollout", zorder=3)
    ax.scatter(pca_mean[0], pca_mean[1], marker="X", s=180, c="crimson",
               zorder=6, label="Goal (pca_mean)")
    ax.scatter(start_pt[0], start_pt[1], marker="*", s=220, c="darkorange",
               edgecolors="black", linewidth=0.8, zorder=7,
               label="Start (nearest to X_pca[0])")
    ev = policy.pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(iter_dir, "latent.png"), dpi=110)
    plt.close()


def save_iteration_artifacts(
    weights,
    iteration,
    result,
    logdir,
    env_cls,
    env_kwargs,
    policy_cls,
    policy_kwargs,
    reward_fn_name,
    termination_fn_name,
    reward_cfg,
    n_components,
    n_bfs,
    video_sim_steps=150,
    video_fps=30,
    video_width=640,
    video_height=360,
):
    """
    Save per-iteration artifacts under <logdir>/iterations/iter_<NNNNNN>/.

    Parameters
    ----------
    weights : np.ndarray
        Flat weight vector for this iteration.
    iteration : int
        Iteration index (used for the folder name).
    result : object
        Whatever the runner gets back from evaluate_single. Should expose
        .total_reward, .distance_x, .terminated, .steps  (props runner does).
    logdir : str
        Training logdir; folder <logdir>/iterations/ is created if needed.
    env_cls, env_kwargs, policy_cls, policy_kwargs :
        Same kwargs used elsewhere to construct env + policy.
    reward_fn_name, termination_fn_name, reward_cfg :
        Forwarded to render_rollout.
    n_components, n_bfs : int
        Used to reshape the weight vector for the heatmap.
    video_sim_steps : int
        How many env steps the recorded video covers. 150 @ 30 fps = 5 s.
    """
    iter_dir = os.path.join(logdir, "iterations", f"iter_{iteration:06d}")
    os.makedirs(iter_dir, exist_ok=True)

    # info.txt — small text summary, useful for grepping later
    info_lines = [
        f"iteration={iteration}",
        f"reward={getattr(result, 'total_reward', float('nan')):.6f}",
        f"distance_x={getattr(result, 'distance_x', float('nan')):.6f}",
        f"terminated={getattr(result, 'terminated', None)}",
        f"steps={getattr(result, 'steps', None)}",
    ]
    with open(os.path.join(iter_dir, "info.txt"), "w") as f:
        f.write("\n".join(info_lines) + "\n")

    try:
        _save_weights_heatmap(weights, iter_dir, n_components, n_bfs)
    except Exception as e:
        print(f"  [heatmap] failed: {e}")

    try:
        _save_latent_plot(weights, iter_dir, policy_cls, policy_kwargs)
    except Exception as e:
        print(f"  [latent] failed: {e}")

    _save_video(
        weights, iter_dir,
        env_cls, env_kwargs, policy_cls, policy_kwargs,
        reward_fn_name, termination_fn_name, reward_cfg,
        sim_steps=video_sim_steps,
        fps=video_fps, width=video_width, height=video_height,
    )


# -----------------------------------------------------------------------
# Post-training reward curve
# -----------------------------------------------------------------------

def _read_training_log(log_path):
    """
    training_log.csv has a yaml-config header prefixed with '#' lines
    before the CSV. Skip those when reading.
    """
    return pd.read_csv(log_path, comment="#")


def save_reward_curve(log_path, out_path, runner_type):
    """
    Produce a reward-vs-time plot from training_log.csv.
    Marks the best iteration with a red star.

    runner_type is one of: 'evolutionary', 'props', 'rl'.
    Picks columns accordingly.
    """
    if not os.path.exists(log_path):
        print(f"[reward curve] log not found: {log_path}")
        return

    try:
        df = _read_training_log(log_path)
    except Exception as e:
        print(f"[reward curve] could not parse {log_path}: {e}")
        return

    if df.empty:
        print(f"[reward curve] log empty: {log_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    if runner_type == "evolutionary":
        x = df["gen"].values
        y = df["best_reward"].values
        ax.plot(x, y, lw=1.5, label="best of generation")
        if "overall_best" in df.columns:
            ax.plot(x, df["overall_best"].values, lw=1.2, alpha=0.7,
                    label="overall best so far")
        if "avg_reward" in df.columns:
            ax.plot(x, df["avg_reward"].values, lw=1.0, alpha=0.5,
                    label="avg of generation")
        x_label = "generation"

    elif runner_type == "props":
        # Drop warmup rows for the main curve so the optimization phase
        # is the focus.
        if "phase" in df.columns:
            df_optim = df[df["phase"] == "optim"]
            df_warmup = df[df["phase"] == "warmup"]
        else:
            df_optim = df
            df_warmup = df.iloc[0:0]
        x = df_optim["iteration"].values
        y = df_optim["reward"].values
        ax.plot(x, y, lw=1.2, label="iteration reward")
        if "overall_best_reward" in df_optim.columns:
            ax.plot(x, df_optim["overall_best_reward"].values,
                    lw=1.5, label="overall best so far")
        if not df_warmup.empty:
            ax.scatter(df_warmup["iteration"].values,
                       df_warmup["reward"].values,
                       s=15, c="gray", alpha=0.6, label="warmup")
        x_label = "iteration"

    else:
        # Generic fallback: just plot a column named 'reward' if present
        if "reward" in df.columns:
            y = df["reward"].values
            x = np.arange(len(y))
            ax.plot(x, y, lw=1.2, label="reward")
        else:
            print(f"[reward curve] don't know how to plot runner '{runner_type}'")
            plt.close()
            return
        x_label = "step"

    # Star the best point
    if len(y) > 0:
        best_i = int(np.argmax(y))
        ax.scatter([x[best_i]], [y[best_i]], marker="*", s=260, c="red",
                   edgecolors="black", linewidth=0.8, zorder=10,
                   label=f"best (reward={y[best_i]:.2f})")

    ax.set_xlabel(x_label)
    ax.set_ylabel("reward")
    ax.set_title("Training reward curve")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[reward curve] saved: {out_path}")