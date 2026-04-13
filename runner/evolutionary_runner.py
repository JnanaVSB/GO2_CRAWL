"""
Evolutionary Runner (CMA-ES).

Training loop for evolutionary DMP weight optimization:

    1. Agent asks for candidate weight vectors
    2. evaluate_single() converts each to a trajectory and steps the env
    3. Agent receives rewards and updates

Rollout logic, logging, and plotting are imported from evaluation/.
"""

import os
import time
import datetime
import numpy as np
import multiprocessing as mp

from evaluation.rollout import evaluate_single, build_eval_args
from evaluation.logging import write_log_header, save_pca_plot, setup_logdir


LOG_COLUMNS = (
    "timestamp,gen,best_reward,avg_reward,overall_best,"
    "best_dist_x,avg_dist_x,time_sec"
)


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
    checkpoint_dir, plots_dir = setup_logdir(logdir)

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

    # Log file
    log_path = os.path.join(logdir, "training_log.csv")
    write_log_header(log_path, full_config or {}, LOG_COLUMNS)
    log_file = open(log_path, "a")

    print(f"Starting training: generations {start_gen} to {max_generations}")

    try:
        for gen in range(start_gen, max_generations):
            gen_start = time.time()

            candidates = agent.ask()

            eval_args = [
                build_eval_args(
                    c, env_cls, env_kwargs, policy_cls, policy_kwargs,
                    sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
                )
                for c in candidates
            ]

            if n_workers > 1:
                with mp.Pool(processes=n_workers) as pool:
                    results = pool.map(evaluate_single, eval_args)
            else:
                results = [evaluate_single(a) for a in eval_args]

            rewards = [r.total_reward for r in results]
            distances = [r.distance_x for r in results]

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

                save_pca_plot(
                    policy_cls, policy_kwargs,
                    overall_best_weights, f"Gen {gen}",
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

    save_pca_plot(
        policy_cls, policy_kwargs,
        best_w, "Final", best_r,
        dist_history[-1] if dist_history else 0.0,
        os.path.join(plots_dir, "pca_final.png"),
    )

    print(f"Training complete. Best reward: {best_r:.4f}")