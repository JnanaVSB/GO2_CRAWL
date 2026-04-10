"""
ProPS-DMP Runner.

Training loop for LLM-based DMP weight optimization (ProPS / ProPS+).

Same role as the other runners:
    1. Warmup: evaluate random weight vectors to seed the LLM history
    2. For each iteration:
       a. Agent builds prompt from history, queries LLM
       b. LLM proposes new weight vector + reasoning
       c. Runner evaluates via rollout (same _evaluate_single as others)
       d. Result stored in agent's history buffer
       e. Prompt + response + reasoning saved to disk
    3. Logging, checkpointing, PCA plots match the other runners
       so that visualize.py works without modification.

Key difference from other runners:
    - No gradient updates — the LLM IS the optimizer.
    - Each iteration involves one LLM API call (seconds, not milliseconds).
    - Prompt and reasoning are saved per-iteration for analysis.
    - Retry logic on LLM parse failures (re-query up to max_parse_retries).
"""

import os
import time
import datetime
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.rewards import get_reward_fn, get_termination_fn


# ---------------------------------------------------------------------------
# Rollout — same logic as evolutionary_runner._evaluate_single
# ---------------------------------------------------------------------------

def _evaluate_single(args):
    """
    Evaluate one DMP weight vector via MuJoCo rollout.
    Creates its own env and policy (for multiprocessing compatibility).

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


def _get_pca_range(policy_cls, policy_kwargs, weights):
    """
    Compute PCA latent trajectory range for a given weight vector.
    Used for the 'detailed' feedback mode.

    Returns
    -------
    pca_x_range : tuple (min, max)
    pca_y_range : tuple (min, max)
    """
    policy = policy_cls(**policy_kwargs)
    latent_traj, _ = policy.generate_latent_trajectory(weights)
    pca_x_range = (float(latent_traj[:, 0].min()), float(latent_traj[:, 0].max()))
    pca_y_range = (float(latent_traj[:, 1].min()), float(latent_traj[:, 1].max()))
    return pca_x_range, pca_y_range


# ---------------------------------------------------------------------------
# PCA plot — same as other runners
# ---------------------------------------------------------------------------

def _save_pca_plot(policy_cls, policy_kwargs, weights, iteration, reward, dist_x, save_path):
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
    plt.title(f"Iter {iteration} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
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
            "timestamp,iteration,phase,reward,cost,avg_reward_20,"
            "overall_best_reward,dist_x,avg_dist_x_20,"
            "api_time_sec,total_time_sec\n"
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
    total_iterations,
    sim_steps,
    reward_fn_name,
    termination_fn_name,
    reward_cfg,
    logdir,
    n_workers,
    checkpoint_every,
    log_every,
    max_parse_retries=3,
    full_config=None,
):
    """
    ProPS/ProPS+ training loop for DMP weight optimization.

    Two phases:
        1. Warmup: evaluate agent.warmup_episodes random weight vectors,
           store in history buffer. No LLM calls.
        2. Optimization: for each iteration, the LLM proposes weights,
           runner evaluates, stores result. Prompt + reasoning saved.

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
    agent : ProPSDMPAgent
        ProPS agent with propose_weights / store_result / generate_random_weights.
    total_iterations : int
        Total LLM optimization iterations (not counting warmup).
    sim_steps : int
        Simulation steps per rollout.
    reward_fn_name : str
        Name of reward function in REWARD_REGISTRY.
    termination_fn_name : str
        Name of termination function in TERMINATION_REGISTRY.
    reward_cfg : dict
        Reward/termination config parameters.
    logdir : str
        Directory for logs, checkpoints, plots, reasoning.
    n_workers : int
        Reserved for future use (evaluation is sequential).
    checkpoint_every : int
        Save checkpoint every N iterations.
    log_every : int
        Print and log every N iterations.
    max_parse_retries : int
        Max times to re-query the LLM if response parsing fails.
    full_config : dict or None
        Full config dict for log header.
    """
    os.makedirs(logdir, exist_ok=True)
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    plots_dir = os.path.join(logdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    reasoning_dir = os.path.join(logdir, "reasoning")
    os.makedirs(reasoning_dir, exist_ok=True)

    # --- Resume from checkpoint ---
    latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest")
    start_iteration = 0
    reward_history = []
    dist_history = []
    cost_history = []

    if os.path.exists(latest_ckpt + "_meta.npz"):
        agent.load(latest_ckpt)
        start_iteration = agent.iterations_done

        hist_path = os.path.join(checkpoint_dir, "histories.npz")
        if os.path.exists(hist_path):
            hist = np.load(hist_path, allow_pickle=True)
            reward_history = hist["reward_history"].tolist()
            dist_history = hist["dist_history"].tolist()
            if "cost_history" in hist:
                cost_history = hist["cost_history"].tolist()

        print(f"Resumed from iteration {start_iteration}")

    # --- Log file ---
    log_path = os.path.join(logdir, "training_log.csv")
    _write_log_header(log_path, full_config or {})
    log_file = open(log_path, "a")

    # Shared eval args builder
    def _build_eval_args(weights):
        return (
            weights, env_cls, env_kwargs, policy_cls, policy_kwargs,
            sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
        )

    # Determine if we need detailed feedback
    detailed = agent.feedback_mode == "detailed"

    # =====================================================================
    # Phase 1: Warmup (random rollouts to seed history)
    # =====================================================================

    warmup_needed = max(0, agent.warmup_episodes - agent.buffer.size())
    if warmup_needed > 0 and start_iteration == 0:
        print(f"Warmup: evaluating {warmup_needed} random weight vectors...")

        for w_idx in range(warmup_needed):
            w_start = time.time()

            weights = agent.generate_random_weights()
            total_reward, steps, terminated, distance_x = _evaluate_single(
                _build_eval_args(weights),
            )

            # Build metadata for detailed mode
            metadata = {
                "distance_x": distance_x,
                "terminated": terminated,
                "steps": steps,
            }
            if detailed:
                pca_x_range, pca_y_range = _get_pca_range(
                    policy_cls, policy_kwargs, weights,
                )
                metadata["pca_x_range"] = pca_x_range
                metadata["pca_y_range"] = pca_y_range

            agent.store_result(weights, total_reward, metadata)
            agent.update_best(weights, total_reward)

            cost = -total_reward
            reward_history.append(total_reward)
            dist_history.append(distance_x)
            cost_history.append(cost)

            w_time = time.time() - w_start
            now = datetime.datetime.now().isoformat()

            print(
                f"  Warmup {w_idx + 1}/{warmup_needed}: "
                f"reward={total_reward:.4f}, cost={cost:.4f}, "
                f"dist_x={distance_x:.4f}, time={w_time:.1f}s"
            )
            log_file.write(
                f"{now},{w_idx},warmup,{total_reward:.6f},{cost:.6f},"
                f"{np.mean(reward_history[-20:]):.6f},"
                f"{agent.best_reward:.6f},{distance_x:.6f},"
                f"{np.mean(dist_history[-20:]):.6f},"
                f"0.00,{w_time:.2f}\n"
            )
            log_file.flush()

        print(f"Warmup complete. Buffer size: {agent.buffer.size()}")

    # =====================================================================
    # Phase 2: LLM optimization
    # =====================================================================

    print(
        f"Starting ProPS optimization: iterations {start_iteration} "
        f"to {total_iterations}"
    )

    try:
        for it in range(start_iteration, total_iterations):
            it_start = time.time()

            # 1. LLM proposes weights (with parse retry)
            weights = None
            reasoning = ""
            full_prompt = ""
            raw_response = ""
            api_time = 0.0

            for parse_attempt in range(max_parse_retries):
                try:
                    (weights, reasoning, full_prompt,
                     raw_response, api_time) = agent.propose_weights()
                    break
                except ValueError as e:
                    print(
                        f"  Parse attempt {parse_attempt + 1}/"
                        f"{max_parse_retries} failed: {e}"
                    )
                    if parse_attempt == max_parse_retries - 1:
                        # All retries failed — use random weights as fallback
                        print("  All parse retries failed. Using random weights.")
                        weights = agent.generate_random_weights()
                        reasoning = "PARSE_FAILED: fell back to random weights"

            # Runner owns the iteration counter — one increment per logical iteration
            agent.iterations_done += 1

            # 2. Evaluate via MuJoCo rollout
            total_reward, steps, terminated, distance_x = _evaluate_single(
                _build_eval_args(weights),
            )

            # 3. Build metadata
            metadata = {
                "distance_x": distance_x,
                "terminated": terminated,
                "steps": steps,
            }
            if detailed:
                pca_x_range, pca_y_range = _get_pca_range(
                    policy_cls, policy_kwargs, weights,
                )
                metadata["pca_x_range"] = pca_x_range
                metadata["pca_y_range"] = pca_y_range

            # 4. Store result in agent's history buffer
            agent.store_result(weights, total_reward, metadata)

            # 5. Track best
            agent.update_best(weights, total_reward)

            cost = -total_reward
            reward_history.append(total_reward)
            dist_history.append(distance_x)
            cost_history.append(cost)

            it_time = time.time() - it_start
            now = datetime.datetime.now().isoformat()

            # --- Save reasoning to disk (every iteration) ---
            reasoning_file = os.path.join(
                reasoning_dir, f"iter_{it:06d}.txt",
            )
            with open(reasoning_file, "w") as rf:
                rf.write("=" * 60 + "\n")
                rf.write(f"Iteration: {it}\n")
                rf.write(f"Timestamp: {now}\n")
                rf.write(f"Reward: {total_reward:.6f}\n")
                rf.write(f"Cost: {cost:.6f}\n")
                rf.write(f"Distance X: {distance_x:.6f}\n")
                rf.write(f"Terminated: {terminated}\n")
                rf.write(f"Steps: {steps}\n")
                rf.write(f"API Time: {api_time:.2f}s\n")
                rf.write("=" * 60 + "\n\n")
                rf.write("PROMPT:\n")
                rf.write("-" * 60 + "\n")
                rf.write(full_prompt)
                rf.write("\n\n")
                rf.write("RAW RESPONSE:\n")
                rf.write("-" * 60 + "\n")
                rf.write(raw_response)
                rf.write("\n\n")
                rf.write("PARSED WEIGHTS:\n")
                rf.write("-" * 60 + "\n")
                for i, wv in enumerate(weights):
                    rf.write(f"w[{i}]: {wv:.6f}\n")
                rf.write("\n")
                rf.write("REASONING:\n")
                rf.write("-" * 60 + "\n")
                rf.write(reasoning)
                rf.write("\n")

            # --- Logging ---
            if it % log_every == 0:
                best_weights, best_reward = agent.get_best()
                avg_reward_20 = np.mean(reward_history[-20:])
                avg_dist_20 = np.mean(dist_history[-20:])

                print(
                    f"Iter {it}: "
                    f"reward={total_reward:.4f}, "
                    f"cost={cost:.4f}, "
                    f"avg_20={avg_reward_20:.4f}, "
                    f"best={best_reward:.4f}, "
                    f"dist_x={distance_x:.4f}, "
                    f"api={api_time:.1f}s, "
                    f"total={it_time:.1f}s"
                )
                log_file.write(
                    f"{now},{it},optim,{total_reward:.6f},{cost:.6f},"
                    f"{avg_reward_20:.6f},{best_reward:.6f},"
                    f"{distance_x:.6f},{avg_dist_20:.6f},"
                    f"{api_time:.2f},{it_time:.2f}\n"
                )
                log_file.flush()

            # --- Checkpointing ---
            if it > 0 and it % checkpoint_every == 0:
                # Numbered checkpoint
                numbered_ckpt = os.path.join(
                    checkpoint_dir, f"checkpoint_iter_{it:06d}",
                )
                agent.save(numbered_ckpt)

                # Latest checkpoint (for resume)
                agent.save(latest_ckpt)

                # Histories
                np.savez(
                    os.path.join(checkpoint_dir, "histories.npz"),
                    reward_history=np.array(reward_history),
                    dist_history=np.array(dist_history),
                    cost_history=np.array(cost_history),
                )

                # Best weights (for visualize.py)
                best_weights, best_reward = agent.get_best()
                if best_weights is not None:
                    np.save(
                        os.path.join(logdir, "best_weights.npy"),
                        best_weights,
                    )

                # PCA trajectory plot
                if best_weights is not None:
                    _save_pca_plot(
                        policy_cls, policy_kwargs,
                        best_weights, it,
                        best_reward, distance_x,
                        os.path.join(plots_dir, f"pca_iter_{it:06d}.png"),
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
        cost_history=np.array(cost_history),
    )

    best_w, best_r = agent.get_best()
    if best_w is not None:
        np.save(os.path.join(logdir, "best_weights.npy"), best_w)

        _save_pca_plot(
            policy_cls, policy_kwargs,
            best_w, it if 'it' in dir() else 0,
            best_r,
            dist_history[-1] if dist_history else 0.0,
            os.path.join(plots_dir, "pca_final.png"),
        )

    print(
        f"Training complete. Best reward: {best_r:.4f}, "
        f"Best cost: {-best_r:.4f}, "
        f"Total API time: {agent.total_api_time:.1f}s"
    )