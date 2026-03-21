import numpy as np
import sys
import os
sys.path.append("/home/jnana/ARLTask/Go2")
import mujoco
import mujoco.viewer
from crawl_env import CrawlEvaluator

XML_PATH = "/home/jnana/ARLTask/Go2/go2/scene.xml"
CSV_PATH = "/home/jnana/ARLTask/Go2/new_dataset_v1.csv"
DMP_TIMESTEPS = 628

# ========================================
# Config for each bfs setting
# ========================================
BFS_CONFIGS = {
    "1": {
        "title": "10-BFS (20 params)",
        "n_bfs": 10,
        "dmp_params": "/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs10.npz",
        "weights": "/home/jnana/ARLTask/Go2_crawl/best_weights_from_checkpoint.npy",
    },
    "2": {
        "title": "15-BFS (30 params)",
        "n_bfs": 15,
        "dmp_params": "/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs15.npz",
        "weights": "/home/jnana/ARLTask/Go2_crawl/best_weights_from_checkpoint_bfs15.npy",
    },
    "3": {
        "title": "20-BFS (40 params)",
        "n_bfs": 20,
        "dmp_params": "/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs20.npz",
        "weights": "/home/jnana/ARLTask/Go2_crawl/best_weights_from_checkpoint_bfs20.npy",
    },
}


def visualize_weights(weights_flat, title, n_bfs, dmp_params_path):
    env = CrawlEvaluator(
        xml_path=XML_PATH,
        csv_path=CSV_PATH,
        dmp_params_path=dmp_params_path,
        n_bfs=n_bfs,
        sim_steps=1000,
        dmp_timesteps=DMP_TIMESTEPS,
    )

    print(f"\n{'='*60}")
    print(f"Evaluating: {title}")
    print(f"n_bfs={n_bfs}, params={n_bfs * 2}")
    print(f"{'='*60}")

    reward, info = env.evaluate_weights(weights_flat, verbose=True)

    print(f"\nNow visualizing in MuJoCo viewer...")
    print(f"Close the viewer window when done.\n")

    env.reset()
    latent_traj, joint_traj = env.build_trajectory_from_weights(weights_flat)

    trajectory_idx = 0
    step_counter = 0
    steps_per_point = 4
    loop_count = 0

    start_x = float(env.data.qpos[0])

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            target = joint_traj[trajectory_idx]

            for j in range(12):
                q = env.data.qpos[env.qpos_index[j]]
                qd = env.data.qvel[env.qvel_index[j]]
                gc = env.data.qfrc_bias[env.qvel_index[j]]
                tau = env.kp * (target[j] - q) + env.kd * (0.0 - qd) + gc
                env.data.ctrl[j] = np.clip(tau, env.ctrl_lo[j], env.ctrl_hi[j])

            mujoco.mj_step(env.model, env.data)
            viewer.sync()

            step_counter += 1
            if step_counter >= steps_per_point:
                step_counter = 0
                trajectory_idx += 1

                if trajectory_idx >= len(joint_traj):
                    trajectory_idx = 0
                    loop_count += 1
                    dist_x = float(env.data.qpos[0]) - start_x
                    height = float(env.data.qpos[2])
                    print(f"Loop {loop_count}: dist_x={dist_x:.4f}, height={height:.4f}")


if __name__ == "__main__":
    print("Which configuration to evaluate?")
    print("  1: 10-BFS (20 parameters)")
    print("  2: 15-BFS (30 parameters)")
    print("  3: 20-BFS (40 parameters)")
    print("  4: Custom weights file")

    choice = input("Enter choice: ").strip()

    if choice in BFS_CONFIGS:
        cfg = BFS_CONFIGS[choice]
        path = cfg["weights"]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            print("Update the path in BFS_CONFIGS.")
            sys.exit()
        weights = np.load(path).flatten()
        print(f"Loaded weights shape: {weights.shape}")
        visualize_weights(weights, cfg["title"], cfg["n_bfs"], cfg["dmp_params"])

    elif choice == "4":
        path = input("Weights file path: ").strip()
        n_bfs = int(input("n_bfs (10/15/20): ").strip())
        dmp_path = input("DMP params path: ").strip()
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit()
        weights = np.load(path).flatten()
        print(f"Loaded weights shape: {weights.shape}")
        visualize_weights(weights, f"Custom ({n_bfs} bfs)", n_bfs, dmp_path)

    else:
        print("Invalid choice")