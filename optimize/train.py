import os
import sys
import numpy as np
import multiprocessing as mp

sys.path.append("/home/jnana/ARLTask/G02_CRAWL")

from openaies import OPENAIES
from crawl_env import CrawlEvaluator

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Config

SAVE_DIR = "/home/jnana/ARLTask/Go2_crawl/results/openaies_result_newV1_1"
PLOTS_DIR = os.path.join(SAVE_DIR, "plots")
CHECKPOINT_DIR = os.path.join(SAVE_DIR, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.npz")
BEST_WEIGHTS_PATH = os.path.join(SAVE_DIR, "best_weights_newV1.npy")
LOG_PATH = os.path.join(SAVE_DIR, "training_log.csv")

XML_PATH = "/home/jnana/ARLTask/Go2_crawl/go2/scene.xml"
CSV_PATH = "/home/jnana/ARLTask/Go2_crawl/dataset/new_dataset_v1.csv"
DMP_PARAMS_PATH = "/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs20.npz"
INITIAL_WEIGHTS_PATH = "/home/jnana/ARLTask/Go2_crawl/weights and params/initial_dmp_weights_newV1_bfs20.npy

N_BFS = 10
SIM_STEPS = 1000
DMP_TIMESTEPS = 628

N_GENERATIONS = 2000
N_WORKERS = 4
CHECKPOINT_EVERY = 100
LOG_EVERY = 10
PLOT_EVERY = 100

LEARNING_RATE = 0.01
NOISE_STD = 0.8
POPULATION_SIZE = 20

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Main-process env

env = CrawlEvaluator(
    xml_path=XML_PATH,
    csv_path=CSV_PATH,
    n_bfs=N_BFS,
    sim_steps=SIM_STEPS,
    dmp_timesteps=DMP_TIMESTEPS,
    dmp_params_path=DMP_PARAMS_PATH,
)


# Worker env

_worker_env = None


def init_worker():
    global _worker_env
    _worker_env = CrawlEvaluator(
        xml_path=XML_PATH,
        csv_path=CSV_PATH,
        n_bfs=N_BFS,
        sim_steps=SIM_STEPS,
        dmp_timesteps=DMP_TIMESTEPS,
        dmp_params_path=DMP_PARAMS_PATH,
    )


def eval_candidate(candidate):
    global _worker_env
    reward, info = _worker_env.evaluate_weights(np.array(candidate), verbose=False)
    return reward, info


def save_pca_plot(env, weights, gen, reward, dist_x):
    latent_traj, joint_traj = env.build_trajectory_from_weights(weights)

    plt.figure(figsize=(8, 8))
    plt.scatter(env.X_pca[:, 0], env.X_pca[:, 1], s=60, c="blue", label="Dataset poses")
    for i, lbl in enumerate(env.labels):
        plt.text(env.X_pca[i, 0] + 0.02, env.X_pca[i, 1] + 0.02, lbl, fontsize=7)

    plt.plot(latent_traj[:, 0], latent_traj[:, 1], "r-", linewidth=2, label="Current best trajectory")
    plt.scatter(latent_traj[0, 0], latent_traj[0, 1], s=100, c="orange", label="Start", zorder=6)

    plt.xlabel(f"PC1 ({100 * env.pca.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100 * env.pca.explained_variance_ratio_[1]:.1f}%)")
    plt.title(f"Gen {gen} | Reward {reward:.1f} | dist_x {dist_x:.4f}")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"gen_{gen:04d}.png"), dpi=100)
    plt.close()


def save_checkpoint(es, gen, best_reward, best_weights):
    ckpt_gen_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_gen_{gen:04d}.npz")
    np.savez(
        ckpt_gen_path,
        generation=gen,
        es_weights=es.weights,
        es_m=es.m,
        es_v=es.v,
        es_t=es.t,
        best_reward=best_reward,
        best_weights=best_weights,
    )

    np.savez(
        CHECKPOINT_PATH,
        generation=gen,
        es_weights=es.weights,
        es_m=es.m,
        es_v=es.v,
        es_t=es.t,
        best_reward=best_reward,
        best_weights=best_weights,
    )

    np.save(BEST_WEIGHTS_PATH, best_weights)


def main():
    log_exists = os.path.exists(LOG_PATH)

    n_params = env.num_params
    es = OPENAIES(
        n_params=n_params,
        learning_rate=LEARNING_RATE,
        noise_std=NOISE_STD,
        population_size=POPULATION_SIZE,
    )

    start_gen = 0
    best_reward = -np.inf
    best_weights = None

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = np.load(CHECKPOINT_PATH)
        es.weights = ckpt["es_weights"]
        es.m = ckpt["es_m"]
        es.v = ckpt["es_v"]
        es.t = int(ckpt["es_t"])
        start_gen = int(ckpt["generation"]) + 1
        best_reward = float(ckpt["best_reward"])
        best_weights = ckpt["best_weights"]
        print(f"Resumed from checkpoint at generation {start_gen}")
        log_mode = "a"
    else:
        initial_weights = np.load(INITIAL_WEIGHTS_PATH).flatten()
        es.set_weights(initial_weights)
        best_weights = initial_weights.copy()
        print("Starting from initial weights")
        log_mode = "w"

    print(f"Starting optimization: {n_params} parameters, {N_GENERATIONS} generations")
    print(f"Workers: {N_WORKERS}")

    with open(LOG_PATH, log_mode) as log_file, mp.Pool(processes=N_WORKERS, initializer=init_worker) as pool:
        if log_mode == "w" or not log_exists:
            log_file.write("gen,best_reward,avg_reward,overall_best,dist_x,height,steps,terminated\n")
            log_file.flush()

        try:
            for gen in range(start_gen, N_GENERATIONS):
                candidates = es.ask()

                results = pool.map(eval_candidate, candidates)
                rewards = [r[0] for r in results]
                infos = [r[1] for r in results]

                es.tell(rewards)

                gen_best_idx = int(np.argmax(rewards))
                gen_best_reward = rewards[gen_best_idx]
                gen_avg_reward = float(np.mean(rewards))
                gen_best_info = infos[gen_best_idx]

                if gen_best_reward > best_reward:
                    best_reward = gen_best_reward
                    best_weights = np.array(candidates[gen_best_idx]).copy()
                    np.save(BEST_WEIGHTS_PATH, best_weights)

                best_info = None
                if gen % LOG_EVERY == 0 or gen % PLOT_EVERY == 0:
                    _, best_info = env.evaluate_weights(best_weights, verbose=False)

                if gen % LOG_EVERY == 0:
                    total_steps = best_info.get("sim_steps_completed", 0)
                    print(
                        f"Gen {gen}: best={gen_best_reward:.4f}, "
                        f"avg={gen_avg_reward:.4f}, "
                        f"overall_best={best_reward:.4f}, "
                        f"dist_x={best_info['distance_x']:.4f}, "
                        f"height={best_info['final_height']:.4f}, "
                        f"steps={total_steps}, "
                        f"term={best_info['terminated']}"
                    )

                    log_file.write(
                        f"{gen},{gen_best_reward:.6f},{gen_avg_reward:.6f},{best_reward:.6f},"
                        f"{best_info['distance_x']:.6f},{best_info['final_height']:.6f},"
                        f"{best_info['sim_steps_completed']},{best_info['terminated']}\n"
                    )
                    log_file.flush()

                if gen % PLOT_EVERY == 0:
                    save_pca_plot(env, best_weights, gen, best_reward, best_info["distance_x"])
                    print(f"  Saved PCA plot for gen {gen}")

                if gen > 0 and gen % CHECKPOINT_EVERY == 0:
                    save_checkpoint(es, gen, best_reward, best_weights)
                    print(f"  Saved checkpoint at generation {gen}")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user!")
            print("Saving emergency checkpoint before exit...")
            save_checkpoint(es, gen, best_reward, best_weights)

    print("\nOptimization complete!")
    print(f"Best reward: {best_reward:.4f}")

    save_checkpoint(es, gen, best_reward, best_weights)

    print("\nFinal evaluation:")
    env.evaluate_weights(best_weights, verbose=True)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()

