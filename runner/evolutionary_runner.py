"""
Evolutionary Runner.

Connects env + policy + agent + reward/termination:

    1. Agent asks for candidate weight vectors
    2. Policy converts each candidate to a joint trajectory
    3. Runner steps env, computes reward and termination externally
    4. Agent receives rewards and updates
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


def _evaluate_single(args):
    """
    Worker function for parallel evaluation.
    Creates its own env and policy per process.
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


def _save_pca_plot(policy_cls, policy_kwargs, weights, gen, reward, dist_x, save_path):
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
    plt.title(f"Gen {gen} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def _write_log_header(log_path, config):
    """Write config header and CSV columns to log file."""
    with open(log_path, "a") as f:
        f.write("\n")
        f.write("#" * 60 + "\n")
        f.write(f"# Run started: {datetime.datetime.now().isoformat()}\n")
        f.write("#" * 60 + "\n")
        f.write(yaml.dump(config, default_flow_style=False))
        f.write("#" * 60 + "\n")
        f.write("timestamp,gen,best_reward,avg_reward,overall_best,best_dist_x,avg_dist_x,time_sec\n")


def run_training_loop(
    env_cls,
    env_kwargs,
    policy_cls,
    policy_kwargs,
    agent,
    max_generations,
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
    os.makedirs(logdir, exist_ok=True)
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(logdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Check for existing checkpoint
    latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.npz")
    start_gen = 0
    reward_history = []
    dist_history = []

    if os.path.exists(latest_ckpt):
        ckpt = np.load(latest_ckpt, allow_pickle=True)
        start_gen = int(ckpt["gen"]) + 1
        reward_history = ckpt["reward_history"].tolist()
        if "dist_history" in ckpt:
            dist_history = ckpt["dist_history"].tolist()
        agent.load_state({
            "es_mean": ckpt["es_mean"],
            "es_sigma": float(ckpt["es_sigma"]),
            "best_weights": ckpt["best_weights"],
            "best_reward": float(ckpt["best_reward"]),
        })
        print(f"Resumed from generation {start_gen}")

    # Log file — always append, write header for this run
    log_path = os.path.join(logdir, "training_log.csv")
    _write_log_header(log_path, full_config or {})
    log_file = open(log_path, "a")

    print(f"Starting training: generations {start_gen} to {max_generations}")

    try:
        for gen in range(start_gen, max_generations):
            gen_start = time.time()

            candidates = agent.ask()

            eval_args = [
                (c, env_cls, env_kwargs, policy_cls, policy_kwargs,
                 sim_steps, reward_fn_name, termination_fn_name, reward_cfg)
                for c in candidates
            ]

            if n_workers > 1:
                with mp.Pool(processes=n_workers) as pool:
                    results = pool.map(_evaluate_single, eval_args)
            else:
                results = [_evaluate_single(a) for a in eval_args]

            rewards = [r[0] for r in results]
            distances = [r[3] for r in results]

            agent.tell(candidates, rewards)

            gen_best_idx = np.argmax(rewards)
            gen_best = rewards[gen_best_idx]
            gen_avg = np.mean(rewards)
            best_dist_x = distances[gen_best_idx]
            avg_dist_x = np.mean(distances)
            overall_best_weights, overall_best_reward = agent.get_best()
            gen_time = time.time() - gen_start
            now = datetime.datetime.now().isoformat()

            reward_history.append(gen_best)
            dist_history.append(best_dist_x)

            if gen % log_every == 0:
                print(
                    f"Gen {gen}: best={gen_best:.4f}, "
                    f"avg={gen_avg:.4f}, "
                    f"overall_best={overall_best_reward:.4f}, "
                    f"dist_x={best_dist_x:.4f}, "
                    f"avg_dist_x={avg_dist_x:.4f}, "
                    f"time={gen_time:.1f}s"
                )
                log_file.write(
                    f"{now},{gen},{gen_best:.6f},{gen_avg:.6f},"
                    f"{overall_best_reward:.6f},{best_dist_x:.6f},"
                    f"{avg_dist_x:.6f},{gen_time:.2f}\n"
                )
                log_file.flush()

            if gen > 0 and gen % checkpoint_every == 0:
                state = agent.get_state()
                np.savez(
                    os.path.join(checkpoint_dir, f"checkpoint_gen_{gen:04d}.npz"),
                    gen=gen,
                    reward_history=np.array(reward_history),
                    dist_history=np.array(dist_history),
                    **state,
                )
                np.savez(
                    latest_ckpt,
                    gen=gen,
                    reward_history=np.array(reward_history),
                    dist_history=np.array(dist_history),
                    **state,
                )
                np.save(
                    os.path.join(logdir, "best_weights.npy"),
                    overall_best_weights,
                )

                # Save PCA trajectory plot
                _save_pca_plot(
                    policy_cls, policy_kwargs,
                    overall_best_weights, gen,
                    overall_best_reward, best_dist_x,
                    os.path.join(plots_dir, f"pca_gen_{gen:04d}.png"),
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    log_file.close()

    state = agent.get_state()
    np.savez(
        latest_ckpt,
        gen=gen,
        reward_history=np.array(reward_history),
        dist_history=np.array(dist_history),
        **state,
    )
    best_w, best_r = agent.get_best()
    np.save(os.path.join(logdir, "best_weights.npy"), best_w)

    # Save final PCA plot
    _save_pca_plot(
        policy_cls, policy_kwargs,
        best_w, gen, best_r,
        dist_history[-1] if dist_history else 0.0,
        os.path.join(plots_dir, "pca_final.png"),
    )

    print(f"Training complete. Best reward: {best_r:.4f}")