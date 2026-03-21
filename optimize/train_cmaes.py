import numpy as np
import sys
import os
import time
import multiprocessing as mp
sys.path.append("/home/jnana/ARLTask/Go2")
import cma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from crawl_env import CrawlEvaluator

SAVE_DIR = "/home/jnana/ARLTask/Go2_crawl/results/cmaes_results_newV1_bfs10"
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "best_weights_cmaes.npy")
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

XML_PATH = "/home/jnana/ARLTask/Go2_crawl/go2/scene.xml"
CSV_PATH = "/home/jnana/ARLTask/Go2_crawl/dataset/new_dataset_v1.csv"
DMP_PARAMS_PATH = "/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs20.npz"
INITIAL_WEIGHTS_PATH = "/home/jnana/ARLTask/Go2_crawl/weights and params/initial_dmp_weights_newV1_bfs20.npy"
N_BFS = 10
SIM_STEPS = 1000
DMP_TIMESTEPS = 628
SIGMA0 = 0.3
POPSIZE = 50
MAX_GENERATIONS = 1000
N_WORKERS = 8
CHECKPOINT_EVERY = 100
LOG_EVERY = 10
PLOT_EVERY = 10

_worker_env = None


def init_worker():
    global _worker_env
    _worker_env = CrawlEvaluator(
        xml_path=XML_PATH,
        csv_path=CSV_PATH,
        dmp_params_path=DMP_PARAMS_PATH,
        n_bfs=N_BFS,
        sim_steps=SIM_STEPS,
        dmp_timesteps=DMP_TIMESTEPS,
    )


def eval_candidate(candidate):
    global _worker_env
    reward, info = _worker_env.evaluate_weights(np.array(candidate), verbose=False)
    return reward, info['distance_x'], info['final_height'], info['sim_steps_completed'], info['terminated']


def save_pca_plot(weights, gen, reward, dist_x):
    env = CrawlEvaluator(
        xml_path=XML_PATH,
        csv_path=CSV_PATH,
        dmp_params_path=DMP_PARAMS_PATH,
        n_bfs=N_BFS,
        sim_steps=SIM_STEPS,
        dmp_timesteps=DMP_TIMESTEPS,
    )
    latent_traj, joint_traj = env.build_trajectory_from_weights(weights)

    plt.figure(figsize=(8, 8))
    plt.scatter(env.X_pca[:, 0], env.X_pca[:, 1], s=60, c="blue", label="Dataset poses")
    for i, lbl in enumerate(env.labels):
        plt.text(env.X_pca[i, 0] + 0.02, env.X_pca[i, 1] + 0.02, lbl, fontsize=7)
    plt.plot(latent_traj[:, 0], latent_traj[:, 1], "r-", linewidth=2, label="Current best trajectory")
    plt.scatter(latent_traj[0, 0], latent_traj[0, 1], s=100, c="orange", label="Start", zorder=6)
    plt.xlabel(f"PC1 ({100 * env.pca.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100 * env.pca.explained_variance_ratio_[1]:.1f}%)")
    plt.title(f"CMA-ES Gen {gen} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "plots", f"gen_{gen:04d}.png"), dpi=100)
    plt.close()


def save_checkpoint(gen, best_reward, best_weights, reward_history, es_mean, es_sigma):
    np.savez(
        os.path.join(CHECKPOINT_DIR, f"checkpoint_gen_{gen:04d}.npz"),
        gen=gen,
        best_reward=best_reward,
        best_weights=best_weights,
        reward_history=np.array(reward_history),
        es_mean=es_mean,
        es_sigma=es_sigma,
    )
    np.savez(
        os.path.join(CHECKPOINT_DIR, "checkpoint_latest.npz"),
        gen=gen,
        best_reward=best_reward,
        best_weights=best_weights,
        reward_history=np.array(reward_history),
        es_mean=es_mean,
        es_sigma=es_sigma,
    )
    np.save(BEST_WEIGHTS_PATH, best_weights)


def main():
    initial_weights = np.load(INITIAL_WEIGHTS_PATH).flatten()
    print(f"Initial weights shape: {initial_weights.shape}")

    # Check for checkpoint to resume
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "checkpoint_latest.npz")
    start_gen = 0
    best_reward = -np.inf
    best_weights = initial_weights.copy()
    reward_history = []
    resume_mean = None
    resume_sigma = None

    if os.path.exists(latest_ckpt_path):
        ckpt = np.load(latest_ckpt_path)
        start_gen = int(ckpt["gen"]) + 1
        best_reward = float(ckpt["best_reward"])
        best_weights = ckpt["best_weights"]
        reward_history = ckpt["reward_history"].tolist()
        resume_mean = ckpt["es_mean"]
        resume_sigma = float(ckpt["es_sigma"])
        print(f"Resumed from checkpoint at generation {start_gen}")
        print(f"Previous best reward: {best_reward:.4f}")
        log_mode = "a"
    else:
        print("Starting from initial weights")
        log_mode = "w"

    # Create CMA-ES
    if resume_mean is not None:
        es = cma.CMAEvolutionStrategy(
            resume_mean.tolist(),
            resume_sigma,
            {
                'popsize': POPSIZE,
                'maxiter': MAX_GENERATIONS,
                'verb_disp': 0,
                'verb_log': 0,
                'tolx': 0,
                'tolfun': 0,
                'tolfunhist': 0,
                'tolstagnation': MAX_GENERATIONS,
                'tolupsigma': 1e20,
                'tolfacupx': 1e20,
            }
        )
    else:
        es = cma.CMAEvolutionStrategy(
            initial_weights.tolist(),
            SIGMA0,
            {
                'popsize': POPSIZE,
                'maxiter': MAX_GENERATIONS,
                'verb_disp': 0,
                'verb_log': 0,
                'tolx': 0,
                'tolfun': 0,
                'tolfunhist': 0,
                'tolstagnation': MAX_GENERATIONS,
                'tolupsigma': 1e20,
                'tolfacupx': 1e20,
            }
        )

    print(f"Starting CMA-ES optimization")
    print(f"Parameters: {len(initial_weights)}, sigma0: {SIGMA0}, popsize: {POPSIZE}")
    print(f"Max generations: {MAX_GENERATIONS}, workers: {N_WORKERS}")

    log_file = open(LOG_PATH, log_mode)
    if log_mode == "w":
        log_file.write("gen,best_reward,avg_reward,overall_best,dist_x,height,steps,terminated,time_sec\n")
        log_file.flush()

    gen = start_gen
    try:
        with mp.Pool(processes=N_WORKERS, initializer=init_worker) as pool:
            for gen in range(start_gen, MAX_GENERATIONS):
                gen_start = time.time()

                if es.stop():
                    reasons = es.stop()
                    print(f"CMA-ES wants to stop at gen {gen}. Reason: {reasons}")
                    print("Continuing anyway...")
                    es.sigma = max(es.sigma, 0.01)

                candidates = es.ask()

                results = pool.map(eval_candidate, [np.array(c) for c in candidates])

                rewards = [r[0] for r in results]
                distances = [r[1] for r in results]
                heights = [r[2] for r in results]
                steps_list = [r[3] for r in results]
                terms = [r[4] for r in results]

                fitness_list = [-r for r in rewards]
                es.tell(candidates, fitness_list)

                gen_best_idx = np.argmax(rewards)
                gen_best_reward = rewards[gen_best_idx]
                gen_avg_reward = np.mean(rewards)
                gen_time = time.time() - gen_start

                reward_history.append(gen_best_reward)

                if gen_best_reward > best_reward:
                    best_reward = gen_best_reward
                    best_weights = np.array(candidates[gen_best_idx]).copy()
                    np.save(BEST_WEIGHTS_PATH, best_weights)

                if gen % LOG_EVERY == 0:
                    print(
                        f"Gen {gen}: best={gen_best_reward:.4f}, "
                        f"avg={gen_avg_reward:.4f}, "
                        f"overall_best={best_reward:.4f}, "
                        f"dist_x={distances[gen_best_idx]:.4f}, "
                        f"height={heights[gen_best_idx]:.4f}, "
                        f"steps={steps_list[gen_best_idx]}, "
                        f"term={terms[gen_best_idx]}, "
                        f"time={gen_time:.1f}s"
                    )
                    log_file.write(
                        f"{gen},{gen_best_reward:.6f},{gen_avg_reward:.6f},"
                        f"{best_reward:.6f},{distances[gen_best_idx]:.6f},"
                        f"{heights[gen_best_idx]:.6f},{steps_list[gen_best_idx]},"
                        f"{terms[gen_best_idx]},{gen_time:.1f}\n"
                    )
                    log_file.flush()

                if gen % PLOT_EVERY == 0:
                    save_pca_plot(best_weights, gen, best_reward, distances[gen_best_idx])
                    print(f"  Saved PCA plot for gen {gen}")

                if gen > 0 and gen % CHECKPOINT_EVERY == 0:
                    save_checkpoint(gen, best_reward, best_weights, reward_history,
                                    np.array(es.mean), es.sigma)
                    print(f"  Saved checkpoint at gen {gen}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user!")
        print("Saving emergency checkpoint...")
        save_checkpoint(gen, best_reward, best_weights, reward_history,
                        np.array(es.mean), es.sigma)

    log_file.close()

    print(f"\nOptimization complete!")
    print(f"Generations: {gen}")
    print(f"Best reward: {best_reward:.4f}")

    save_checkpoint(gen, best_reward, best_weights, reward_history,
                    np.array(es.mean), es.sigma)

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, 'b-', alpha=0.3, linewidth=1)
    window = min(20, max(1, len(reward_history) // 5))
    if window > 1 and len(reward_history) > window:
        running_avg = np.convolve(reward_history, np.ones(window) / window, mode='valid')
        plt.plot(np.arange(window - 1, len(reward_history)), running_avg, 'b-', linewidth=2, label='Running average')
    plt.xlabel('Generation')
    plt.ylabel('Best Reward')
    plt.title('CMA-ES Optimization Progress')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "reward_history.png"), dpi=150)
    plt.close()

    print("\nFinal evaluation:")
    result = eval_candidate(best_weights)
    print(f"Reward: {result[0]:.4f}, dist_x: {result[1]:.4f}, height: {result[2]:.4f}")
    print(f"Reward history plot saved")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

# import numpy as np
# import sys
# import os
# sys.path.append("/home/jnana/ARLTask/Go2")
# import cma
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from crawl_env import CrawlEvaluator

# SAVE_DIR = "cmaes_results_newV1"
# CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
# BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "best_weights_cmaes.npy")
# os.makedirs(SAVE_DIR, exist_ok=True)
# os.makedirs(os.path.join(SAVE_DIR, "plots"), exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# env = CrawlEvaluator(
#     xml_path="/home/jnana/ARLTask/Go2/go2/scene.xml",
#     csv_path="/home/jnana/ARLTask/Go2/new_dataset_v1.csv",
#     n_bfs=10,
#     sim_steps=1000,
#     dmp_timesteps=628
# )

# log_file = open(os.path.join(SAVE_DIR, "training_log.csv"), "w")
# log_file.write("gen,best_reward,avg_reward,overall_best,dist_x,height,steps,terminated\n")

# initial_weights = np.load("/home/jnana/ARLTask/Go2/initial_dmp_weights_newV0.npy").flatten()
# print(f"Initial weights shape: {initial_weights.shape}")


# def save_pca_plot(env, weights, gen, reward, dist_x):
#     latent_traj, joint_traj = env.build_trajectory_from_weights(weights)

#     plt.figure(figsize=(8, 8))
#     plt.scatter(env.X_pca[:, 0], env.X_pca[:, 1], s=60, c="blue", label="Dataset poses")
#     for i, lbl in enumerate(env.labels):
#         plt.text(env.X_pca[i, 0] + 0.02, env.X_pca[i, 1] + 0.02, lbl, fontsize=7)

#     plt.plot(latent_traj[:, 0], latent_traj[:, 1], "r-", linewidth=2, label="Current best trajectory")
#     plt.scatter(latent_traj[0, 0], latent_traj[0, 1], s=100, c="orange", label="Start", zorder=6)

#     plt.xlabel(f"PC1 ({100 * env.pca.explained_variance_ratio_[0]:.1f}%)")
#     plt.ylabel(f"PC2 ({100 * env.pca.explained_variance_ratio_[1]:.1f}%)")
#     plt.title(f"CMA-ES Gen {gen} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
#     plt.grid(True, alpha=0.3)
#     plt.axis("equal")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(SAVE_DIR, "plots", f"gen_{gen:04d}.png"), dpi=100)
#     plt.close()


# n_params = env.num_params
# sigma0 = 0.3
# popsize = 50
# max_generations = 5000

# print(f"Starting CMA-ES optimization")
# print(f"Parameters: {n_params}")
# print(f"sigma0: {sigma0}")
# print(f"Population size: {popsize}")
# print(f"Max generations: {max_generations}")

# es = cma.CMAEvolutionStrategy(
#     initial_weights.tolist(),
#     sigma0,
#     {
#         'popsize': popsize,
#         'maxiter': max_generations,
#         'verb_disp': 0,
#         'verb_log': 0,
#         'tolx': 0,
#         'tolfun': 0,
#         'tolfunhist': 0,
#         'tolstagnation': max_generations,
#         'tolupsigma': 1e20,
#         'tolfacupx': 1e20
#     }
# )

# best_reward = -np.inf
# best_weights = initial_weights.copy()
# reward_history = []

# gen = 0
# try:
#     for gen in range(max_generations):
#         if es.stop():
#             reasons = es.stop()
#             print(f"CMA-ES wants to stop at gen {gen}. Reason: {reasons}")
#             print("Continuing anyway...")
#             es.sigma = max(es.sigma, 0.01)

#         candidates = es.ask()

#         fitness_list = []
#         for candidate in candidates:
#             reward, info = env.evaluate_weights(np.array(candidate), verbose=False)
#             fitness_list.append(-reward)

#         es.tell(candidates, fitness_list)

#         gen_rewards = [-f for f in fitness_list]
#         gen_best_idx = np.argmax(gen_rewards)
#         gen_best_reward = gen_rewards[gen_best_idx]
#         gen_avg_reward = np.mean(gen_rewards)

#         reward_history.append(gen_best_reward)

#         if gen_best_reward > best_reward:
#             best_reward = gen_best_reward
#             best_weights = np.array(candidates[gen_best_idx]).copy()
#             np.save(BEST_WEIGHTS_PATH, best_weights)

#         if gen % 10 == 0:
#             _, best_info = env.evaluate_weights(best_weights, verbose=False)
#             print(
#                 f"Gen {gen}: best={gen_best_reward:.4f}, "
#                 f"avg={gen_avg_reward:.4f}, "
#                 f"overall_best={best_reward:.4f}, "
#                 f"dist_x={best_info['distance_x']:.4f}, "
#                 f"height={best_info['final_height']:.4f}, "
#                 f"steps={best_info['sim_steps_completed']}, "
#                 f"term={best_info['terminated']}"
#             )

#             log_file.write(
#                 f"{gen},{gen_best_reward:.6f},{gen_avg_reward:.6f},"
#                 f"{best_reward:.6f},{best_info['distance_x']:.6f},"
#                 f"{best_info['final_height']:.6f},{best_info['sim_steps_completed']},"
#                 f"{best_info['terminated']}\n"
#             )
#             log_file.flush()

#             save_pca_plot(env, best_weights, gen, best_reward, best_info['distance_x'])
#             print(f"  Saved PCA plot for gen {gen}")

#         if gen > 0 and gen % 100 == 0:
#             np.savez(
#                 os.path.join(CHECKPOINT_DIR, f"checkpoint_gen_{gen:04d}.npz"),
#                 gen=gen,
#                 best_reward=best_reward,
#                 best_weights=best_weights,
#                 reward_history=np.array(reward_history),
#             )
#             print(f"  Saved checkpoint at gen {gen}")

# except KeyboardInterrupt:
#     print("\nOptimization interrupted by user!")

# log_file.close()

# print(f"\nOptimization complete!")
# print(f"Generations: {gen}")
# print(f"Best reward: {best_reward:.4f}")

# np.save(BEST_WEIGHTS_PATH, best_weights)
# np.save(os.path.join(SAVE_DIR, "reward_history.npy"), np.array(reward_history))
# np.savez(
#     os.path.join(CHECKPOINT_DIR, f"checkpoint_final.npz"),
#     gen=gen,
#     best_reward=best_reward,
#     best_weights=best_weights,
#     reward_history=np.array(reward_history),
# )
# print(f"Best weights saved to {BEST_WEIGHTS_PATH}")

# print("\nFinal evaluation:")
# env.evaluate_weights(best_weights, verbose=True)

# plt.figure(figsize=(10, 5))
# plt.plot(reward_history, 'b-', alpha=0.3, linewidth=1)
# window = min(20, max(1, len(reward_history) // 5))
# if window > 1 and len(reward_history) > window:
#     running_avg = np.convolve(reward_history, np.ones(window) / window, mode='valid')
#     plt.plot(np.arange(window - 1, len(reward_history)), running_avg, 'b-', linewidth=2, label='Running average')
# plt.xlabel('Generation')
# plt.ylabel('Best Reward')
# plt.title('CMA-ES Optimization Progress')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.savefig(os.path.join(SAVE_DIR, "reward_history.png"), dpi=150)
# plt.close()
# print(f"Reward history plot saved")