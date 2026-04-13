"""
SAC-DMP Runner.

Training loop for SAC-based DMP weight optimization:

    1. Agent samples a DMP weight vector
    2. evaluate_single() converts weights to a trajectory and steps the env
    3. Agent stores transition and runs gradient updates

Rollout logic, logging, and plotting are imported from evaluation/.
"""

import os
import time
import datetime
import numpy as np

from evaluation.rollout import evaluate_single, build_eval_args
from evaluation.logging import write_log_header, save_pca_plot, setup_logdir


LOG_COLUMNS = (
    "timestamp,episode,reward,avg_reward_50,overall_best,"
    "dist_x,avg_dist_x_50,time_sec"
)


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
    SAC training loop for DMP weight optimization.

    Parameters
    ----------
    env_cls : class
        Go2CrawlEnv class.
    env_kwargs : dict
        Constructor kwargs for the env.
    policy_cls : class
        Trajectory generator / policy class.
    policy_kwargs : dict
        Constructor kwargs for the policy.
    agent : SACDMPAgent
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
    checkpoint_dir, plots_dir = setup_logdir(logdir)

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
    write_log_header(log_path, full_config or {}, LOG_COLUMNS)
    log_file = open(log_path, "a")

    print(f"Starting SAC training: episodes {start_episode} to {total_episodes}")

    try:
        for ep in range(start_episode, total_episodes):
            ep_start = time.time()

            # 1. Agent samples DMP weight vector
            obs = agent.get_obs()
            weights = agent.sample_action(obs, deterministic=False)

            # 2. Evaluate: policy converts to trajectory, runner steps env
            eval_args = build_eval_args(
                weights, env_cls, env_kwargs, policy_cls, policy_kwargs,
                sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
            )
            result = evaluate_single(eval_args)

            # 3. Store transition and update
            next_obs = agent.get_obs()
            agent.store_transition(obs, weights, result.total_reward, next_obs, done=True)
            agent.update()

            # 4. Track best
            agent.update_best(weights, result.total_reward)

            reward_history.append(result.total_reward)
            dist_history.append(result.distance_x)

            ep_time = time.time() - ep_start
            now = datetime.datetime.now().isoformat()

            # --- Logging ---
            if ep % log_every == 0:
                best_weights, best_reward = agent.get_best()
                avg_reward_50 = np.mean(reward_history[-50:])
                avg_dist_50 = np.mean(dist_history[-50:])

                print(
                    f"Ep {ep}: "
                    f"reward={result.total_reward:.4f}, "
                    f"avg_50={avg_reward_50:.4f}, "
                    f"best={best_reward:.4f}, "
                    f"dist_x={result.distance_x:.4f}, "
                    f"avg_dist_50={avg_dist_50:.4f}, "
                    f"time={ep_time:.1f}s"
                )
                log_file.write(
                    f"{now},{ep},{result.total_reward:.6f},{avg_reward_50:.6f},"
                    f"{best_reward:.6f},{result.distance_x:.6f},"
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

                # Save best weights as .npy (for visualize.py)
                best_weights, best_reward = agent.get_best()
                if best_weights is not None:
                    np.save(
                        os.path.join(logdir, "best_weights.npy"),
                        best_weights,
                    )

                # PCA trajectory plot
                save_pca_plot(
                    policy_cls, policy_kwargs,
                    best_weights, f"Ep {ep}",
                    best_reward, result.distance_x,
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

        save_pca_plot(
            policy_cls, policy_kwargs,
            best_w, "Final", best_r,
            dist_history[-1] if dist_history else 0.0,
            os.path.join(plots_dir, "pca_final.png"),
        )

    print(f"Training complete. Best reward: {best_r:.4f}")