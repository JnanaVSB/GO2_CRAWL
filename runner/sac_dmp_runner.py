"""
RL Runner for DMP-based policy optimization.

Same role as evolutionary_runner.py but for RL agents (SAC-DMP, PPO-DMP):

    1. Agent samples a DMP weight vector
    2. Policy converts weights to a joint trajectory
    3. Runner steps env, computes reward and termination externally
    4. Agent stores the transition and runs gradient updates

The rollout logic is identical to evolutionary_runner._evaluate_single().
Logging format, checkpointing, and PCA plots match the evolutionary runner
so that visualize.py works without modification.
"""

import os
import time
import datetime
import yaml
import numpy as np
import multiprocessing as mp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.rewards import get_reward_fn, get_termination_fn


# ---------------------------------------------------------------------------
# Rollout (same logic as evolutionary_runner._evaluate_single)
# ---------------------------------------------------------------------------

def _evaluate_single(args):
    """
    Worker function for evaluating one DMP weight vector.
    Creates its own env and policy per process (for multiprocessing).

    Returns
    -------
    total_reward : float
    steps : int
    terminated : bool
    distance_x : float
    """
    (candidate, env_cls, env_kwargs, policy_cls, policy_kwargs,
     sim_steps, reward_fn_name, termination_fn_name, reward_cfg) = args

    env = env_cls(**env_kwargs)
    policy = policy_cls(**policy_kwargs)
    reward_fn = get_reward_fn(reward_fn_name)
    termination_fn = get_termination_fn(termination_fn_name)

    joint_traj = policy.generate_trajectory(candidate)
    traj_len = len(joint_traj)

    prev_obs, _ = env.reset()
    start_x = prev_obs[24]
    total_reward = 0.0
    terminated = False

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

    distance_x = obs[24] - start_x

    return total_reward, t + 1, terminated, distance_x


# ---------------------------------------------------------------------------
# PCA plot (same as evolutionary runner)
# ---------------------------------------------------------------------------

def _save_pca_plot(policy_cls, policy_kwargs, weights, episode, reward, dist_x, save_path):
    """Save PCA trajectory plot for current best weights."""
    policy = policy_cls(**policy_kwargs)
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
    plt.title(f"Ep {episode} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


# ---------------------------------------------------------------------------
# Log header
# ---------------------------------------------------------------------------

def _write_log_header(log_path, config):
    """Write config header and CSV columns to log file."""
    with open(log_path, "a") as f:
        f.write("\n")
        f.write("#" * 60 + "\n")
        f.write(f"# Run started: {datetime.datetime.now().isoformat()}\n")
        f.write("#" * 60 + "\n")
        f.write(yaml.dump(config, default_flow_style=False))
        f.write("#" * 60 + "\n")
        f.write(
            "timestamp,episode,reward,avg_reward_50,overall_best,"
            "dist_x,avg_dist_x_50,time_sec\n"
        )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training_loop(
    env_cls,
    env_kwargs,
    policy_cls,
    policy_kwargs,
    agent,
    total_episodes,
    sim_steps,
    reward_fn_name,
    termination_fn_name,
    reward_cfg,
    logdir,
    n_workers,
    checkpoint_every,
    log_every,
    full_config=None,
):
    """
    RL training loop for DMP weight optimization.

    Parameters
    ----------
    env_cls : class
        Go2CrawlEnv class.
    env_kwargs : dict
        Constructor kwargs for the env.
    policy_cls : class
        DMPPolicy class.
    policy_kwargs : dict
        Constructor kwargs for the policy.
    agent : SACDMPAgent or PPODMPAgent
        RL agent with sample_action / store_transition / update interface.
    total_episodes : int
        Total number of episodes (weight evaluations) to run.
    sim_steps : int
        Simulation steps per episode.
    reward_fn_name : str
        Name of reward function in REWARD_REGISTRY.
    termination_fn_name : str
        Name of termination function in TERMINATION_REGISTRY.
    reward_cfg : dict
        Reward/termination config parameters.
    logdir : str
        Directory for logs, checkpoints, plots.
    n_workers : int
        Number of parallel workers for evaluation.
    checkpoint_every : int
        Save checkpoint every N episodes.
    log_every : int
        Print and log every N episodes.
    full_config : dict or None
        Full config dict for log header.
    """
    os.makedirs(logdir, exist_ok=True)
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(logdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # --- Resume from checkpoint ---
    latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest")
    start_episode = 0
    reward_history = []
    dist_history = []

    # Check for SB3-style checkpoint (agent.save creates {path}.zip)
    if os.path.exists(latest_ckpt + ".zip"):
        agent.load(latest_ckpt)
        start_episode = agent.episodes_done

        # Load histories
        hist_path = os.path.join(checkpoint_dir, "histories.npz")
        if os.path.exists(hist_path):
            hist = np.load(hist_path, allow_pickle=True)
            reward_history = hist["reward_history"].tolist()
            dist_history = hist["dist_history"].tolist()

        print(f"Resumed from episode {start_episode}")

    # --- Log file ---
    log_path = os.path.join(logdir, "training_log.csv")
    _write_log_header(log_path, full_config or {})
    log_file = open(log_path, "a")

    print(f"Starting RL training: episodes {start_episode} to {total_episodes}")

    try:
        for ep in range(start_episode, total_episodes):
            ep_start = time.time()

            # 1. Agent samples DMP weight vector
            obs = agent.get_obs()
            weights = agent.sample_action(obs, deterministic=False)

            # 2. Evaluate: policy converts to trajectory, runner steps env
            eval_args = (
                weights, env_cls, env_kwargs, policy_cls, policy_kwargs,
                sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
            )
            total_reward, steps, terminated, distance_x = _evaluate_single(eval_args)

            # 3. Store transition and update
            next_obs = agent.get_obs()
            agent.store_transition(obs, weights, total_reward, next_obs, done=True)
            agent.update()

            # 4. Track best
            agent.update_best(weights, total_reward)

            reward_history.append(total_reward)
            dist_history.append(distance_x)

            ep_time = time.time() - ep_start
            now = datetime.datetime.now().isoformat()

            # --- Logging ---
            if ep % log_every == 0:
                best_weights, best_reward = agent.get_best()
                avg_reward_50 = np.mean(reward_history[-50:])
                avg_dist_50 = np.mean(dist_history[-50:])

                print(
                    f"Ep {ep}: "
                    f"reward={total_reward:.4f}, "
                    f"avg_50={avg_reward_50:.4f}, "
                    f"best={best_reward:.4f}, "
                    f"dist_x={distance_x:.4f}, "
                    f"avg_dist_50={avg_dist_50:.4f}, "
                    f"time={ep_time:.1f}s"
                )
                log_file.write(
                    f"{now},{ep},{total_reward:.6f},{avg_reward_50:.6f},"
                    f"{best_reward:.6f},{distance_x:.6f},"
                    f"{avg_dist_50:.6f},{ep_time:.2f}\n"
                )
                log_file.flush()

            # --- Checkpointing ---
            if ep > 0 and ep % checkpoint_every == 0:
                # Save numbered checkpoint
                numbered_ckpt = os.path.join(
                    checkpoint_dir, f"checkpoint_ep_{ep:06d}",
                )
                agent.save(numbered_ckpt)

                # Save latest checkpoint (for resume)
                agent.save(latest_ckpt)

                # Save histories (separate from agent checkpoint)
                np.savez(
                    os.path.join(checkpoint_dir, "histories.npz"),
                    reward_history=np.array(reward_history),
                    dist_history=np.array(dist_history),
                )

                # Save best weights as .npy (for visualize.py compatibility)
                best_weights, best_reward = agent.get_best()
                if best_weights is not None:
                    np.save(
                        os.path.join(logdir, "best_weights.npy"),
                        best_weights,
                    )

                # PCA trajectory plot
                _save_pca_plot(
                    policy_cls, policy_kwargs,
                    best_weights, ep,
                    best_reward, distance_x,
                    os.path.join(plots_dir, f"pca_ep_{ep:06d}.png"),
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    log_file.close()

    # --- Final save ---
    agent.save(latest_ckpt)

    np.savez(
        os.path.join(checkpoint_dir, "histories.npz"),
        reward_history=np.array(reward_history),
        dist_history=np.array(dist_history),
    )

    best_w, best_r = agent.get_best()
    if best_w is not None:
        np.save(os.path.join(logdir, "best_weights.npy"), best_w)

        _save_pca_plot(
            policy_cls, policy_kwargs,
            best_w, ep, best_r,
            dist_history[-1] if dist_history else 0.0,
            os.path.join(plots_dir, "pca_final.png"),
        )

    print(f"Training complete. Best reward: {best_r:.4f}")