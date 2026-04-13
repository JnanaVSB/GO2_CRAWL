"""
ProPS / ProPS+ Runner.

Training loop for LLM-based weight optimization:

    1. Warmup: evaluate random weight vectors to seed the LLM history
    2. For each iteration:
       a. Agent builds prompt from history, queries LLM
       b. LLM proposes new weight vector + reasoning
       c. evaluate_single() does the MuJoCo rollout
       d. Result stored in agent's history buffer
       e. Prompt + response + reasoning saved to disk
    3. Logging, checkpointing, PCA plots use shared evaluation/ utilities

Key difference from other runners:
    - No gradient updates — the LLM IS the optimizer.
    - Each iteration involves one LLM API call (seconds, not milliseconds).
    - Prompt and reasoning are saved per-iteration for analysis.
    - Retry logic on LLM parse failures (re-query up to max_parse_retries).

ProPS vs ProPS+ is determined entirely by the config (template and
env_description_file). This runner handles both identically.
"""

import os
import time
import datetime
import numpy as np

from evaluation.rollout import evaluate_single, build_eval_args
from evaluation.logging import (
    write_log_header, save_pca_plot, get_pca_range, setup_logdir,
)


LOG_COLUMNS = (
    "timestamp,iteration,phase,reward,cost,avg_reward_20,"
    "overall_best_reward,dist_x,avg_dist_x_20,"
    "api_time_sec,total_time_sec"
)


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
    ProPS/ProPS+ training loop for weight optimization.

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
        Trajectory generator / policy class.
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
    checkpoint_dir, plots_dir = setup_logdir(logdir)
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
    write_log_header(log_path, full_config or {}, LOG_COLUMNS)
    log_file = open(log_path, "a")

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
            eval_args = build_eval_args(
                weights, env_cls, env_kwargs, policy_cls, policy_kwargs,
                sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
            )
            result = evaluate_single(eval_args)

            # Build metadata
            metadata = {
                "distance_x": result.distance_x,
                "terminated": result.terminated,
                "steps": result.steps,
                "avg_height_dev": result.avg_height_dev,
            }
            if detailed:
                pca_x_range, pca_y_range = get_pca_range(
                    policy_cls, policy_kwargs, weights,
                )
                metadata["pca_x_range"] = pca_x_range
                metadata["pca_y_range"] = pca_y_range

            agent.store_result(
                weights, result.total_reward, result.distance_x,
                result.avg_height_dev, metadata,
            )
            agent.update_best(weights, result.total_reward)

            cost = agent.buffer.entries[-1]["cost"]
            reward_history.append(result.total_reward)
            dist_history.append(result.distance_x)
            cost_history.append(cost)

            w_time = time.time() - w_start
            now = datetime.datetime.now().isoformat()

            print(
                f"  Warmup {w_idx + 1}/{warmup_needed}: "
                f"reward={result.total_reward:.4f}, cost={cost:.4f}, "
                f"dist_x={result.distance_x:.4f}, time={w_time:.1f}s"
            )
            log_file.write(
                f"{now},{w_idx},warmup,{result.total_reward:.6f},{cost:.6f},"
                f"{np.mean(reward_history[-20:]):.6f},"
                f"{agent.best_reward:.6f},{result.distance_x:.6f},"
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
                        print("  All parse retries failed. Using random weights.")
                        weights = agent.generate_random_weights()
                        reasoning = "PARSE_FAILED: fell back to random weights"
                        agent.iterations_done += 1

            # 2. Evaluate via MuJoCo rollout
            eval_args = build_eval_args(
                weights, env_cls, env_kwargs, policy_cls, policy_kwargs,
                sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
            )
            result = evaluate_single(eval_args)

            # 3. Build metadata
            metadata = {
                "distance_x": result.distance_x,
                "terminated": result.terminated,
                "steps": result.steps,
                "avg_height_dev": result.avg_height_dev,
            }
            if detailed:
                pca_x_range, pca_y_range = get_pca_range(
                    policy_cls, policy_kwargs, weights,
                )
                metadata["pca_x_range"] = pca_x_range
                metadata["pca_y_range"] = pca_y_range

            # 4. Store result in agent's history buffer
            agent.store_result(
                weights, result.total_reward, result.distance_x,
                result.avg_height_dev, metadata,
            )

            # 5. Track best
            agent.update_best(weights, result.total_reward)

            cost = agent.buffer.entries[-1]["cost"]
            reward_history.append(result.total_reward)
            dist_history.append(result.distance_x)
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
                rf.write(f"Reward: {result.total_reward:.6f}\n")
                rf.write(f"Cost: {cost:.6f}\n")
                rf.write(f"Distance X: {result.distance_x:.6f}\n")
                rf.write(f"Terminated: {result.terminated}\n")
                rf.write(f"Steps: {result.steps}\n")
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
                    f"reward={result.total_reward:.4f}, "
                    f"cost={cost:.4f}, "
                    f"avg_20={avg_reward_20:.4f}, "
                    f"best={best_reward:.4f}, "
                    f"dist_x={result.distance_x:.4f}, "
                    f"api={api_time:.1f}s, "
                    f"total={it_time:.1f}s"
                )
                log_file.write(
                    f"{now},{it},optim,{result.total_reward:.6f},{cost:.6f},"
                    f"{avg_reward_20:.6f},{best_reward:.6f},"
                    f"{result.distance_x:.6f},{avg_dist_20:.6f},"
                    f"{api_time:.2f},{it_time:.2f}\n"
                )
                log_file.flush()

            # --- Checkpointing ---
            if it > 0 and it % checkpoint_every == 0:
                numbered_ckpt = os.path.join(
                    checkpoint_dir, f"checkpoint_iter_{it:06d}",
                )
                agent.save(numbered_ckpt)

                agent.save(latest_ckpt)

                np.savez(
                    os.path.join(checkpoint_dir, "histories.npz"),
                    reward_history=np.array(reward_history),
                    dist_history=np.array(dist_history),
                    cost_history=np.array(cost_history),
                )

                best_weights, best_reward = agent.get_best()
                if best_weights is not None:
                    np.save(
                        os.path.join(logdir, "best_weights.npy"),
                        best_weights,
                    )

                    save_pca_plot(
                        policy_cls, policy_kwargs,
                        best_weights, f"Iter {it}",
                        best_reward, result.distance_x,
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

        save_pca_plot(
            policy_cls, policy_kwargs,
            best_w, "Final", best_r,
            dist_history[-1] if dist_history else 0.0,
            os.path.join(plots_dir, "pca_final.png"),
        )

    print(
        f"Training complete. Best reward: {best_r:.4f}, "
        f"Best cost: {-best_r:.4f}, "
        f"Total API time: {agent.total_api_time:.1f}s"
    )