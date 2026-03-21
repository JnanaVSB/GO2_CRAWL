# import numpy as np
# import pandas as pd
# import sys
# sys.path.append("/home/jnana/ARLTask/Go2")
# from sklearn.decomposition import PCA
# from dmp.dmp_rhythmic import DMPs_rhythmic

# csv_path = "/home/jnana/ARLTask/Go2/go2_crawl_poses_v7-A.csv"

# df = pd.read_csv(csv_path)
# labels = df.iloc[:, 0].values
# X = df.iloc[:, 1:].values

# print(f"Dataset: {X.shape[0]} poses, {X.shape[1]} joints")

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# pca_mean = np.mean(X_pca, axis=0)
# pca_std = np.std(X_pca, axis=0)

# print(f"PCA explained variance: {pca.explained_variance_ratio_}")
# print(f"PCA mean: {pca_mean}")
# print(f"PCA std: {pca_std}")

# n_bfs = 10
# radius = 0.7

# n_points = 240
# theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
# circle_pca = np.column_stack([
#     pca_mean[0] + radius * np.cos(theta),
#     pca_mean[1] + radius * np.sin(theta),
# ])

# print(f"\nCircle radius: {radius}")
# print(f"Circle shape: {circle_pca.shape}")

# dmp = DMPs_rhythmic(n_dmps=2, n_bfs=n_bfs, ay=np.ones(2) * 10.0)
# dmp.imitate_path(y_des=circle_pca.T)

# print(f"\nDMP trained!")
# print(f"DMP weights shape: {dmp.w.shape}")
# print(f"DMP goal: {dmp.goal}")
# print(f"DMP c shape: {dmp.c.shape}")
# print(f"DMP h shape: {dmp.h.shape}")

# dmp_latent, _, _ = dmp.rollout()
# dmp_joint = pca.inverse_transform(dmp_latent)

# print(f"DMP rollout shape: {dmp_latent.shape}")
# print(f"Joint trajectory shape: {dmp_joint.shape}")

# np.save("initial_dmp_weights_v7_new.npy", dmp.w)

# np.savez("dmp_params_v7.npz",
#     weights=dmp.w,
#     c=dmp.c,
#     h=dmp.h,
#     goal=dmp.goal,
# )

# import pickle
# with open("pca_model.pkl", "wb") as f:
#     pickle.dump(pca, f)

# print(f"\nSaved:")
# print(f"  initial_dmp_weights.npy")
# print(f"  dmp_params.npz (weights, c, h, goal)")
# print(f"  pca_model.pkl")

import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/jnana/ARLTask/Go2")
from sklearn.decomposition import PCA
from dmp.dmp_rhythmic import DMPs_rhythmic


def quat_to_euler_wxyz(q):
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    xml_path = "/home/jnana/ARLTask/Go2/go2/scene.xml"
    csv_path = "/home/jnana/ARLTask/Go2/new_dataset_v1.csv"
    radius = 0.4
    n_bfs = 10
    n_circle_points = 240
    steps_per_point = 4
    kp = 40.0
    kd = 2.0

    # ========================================
    # 1. Load dataset and fit PCA
    # ========================================
    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].values.astype(np.float64)

    print(f"Dataset: {X.shape[0]} poses, {X.shape[1]} joints")
    print(f"Labels: {labels}")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_mean = np.mean(X_pca, axis=0)
    pca_std = np.std(X_pca, axis=0)

    print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total: {sum(pca.explained_variance_ratio_):.3f}")
    print(f"PCA mean: {pca_mean}")
    print(f"PCA std: {pca_std}")
    print(f"PC1 range: {X_pca[:, 0].min():.3f} to {X_pca[:, 0].max():.3f}")
    print(f"PC2 range: {X_pca[:, 1].min():.3f} to {X_pca[:, 1].max():.3f}")

    # ========================================
    # 2. Build circle in PCA space
    # ========================================
    theta = np.linspace(0, 2 * np.pi, n_circle_points, endpoint=False)
    circle_pca = np.column_stack([
        pca_mean[0] + radius * np.cos(theta),
        pca_mean[1] + radius * np.sin(theta),
    ])

    print(f"\nCircle radius: {radius}")
    print(f"Circle shape: {circle_pca.shape}")

    # ========================================
    # 3. Train DMP on circle
    # ========================================
    dmp = DMPs_rhythmic(n_dmps=2, n_bfs=n_bfs, ay=np.ones(2) * 10.0)
    dmp.imitate_path(y_des=circle_pca.T)

    dmp_latent, _, _ = dmp.rollout()
    dmp_joint = pca.inverse_transform(dmp_latent)

    print(f"\nDMP trained!")
    print(f"DMP weights shape: {dmp.w.shape}")
    print(f"DMP goal: {dmp.goal}")
    print(f"DMP rollout shape: {dmp_latent.shape}")
    print(f"Joint trajectory shape: {dmp_joint.shape}")

    # ========================================
    # 4. Save everything
    # ========================================
    np.save("initial_dmp_weights_newV1_bfs10.npy", dmp.w)

    np.savez("dmp_params_newV1_bfs10.npz",
        weights=dmp.w,
        c=dmp.c,
        h=dmp.h,
        goal=dmp.goal,
    )

    import pickle
    with open("pca_model.pkl", "wb") as f:
        pickle.dump(pca, f)

    print(f"\nSaved:")
    print(f"  initial_dmp_weights.npy - shape {dmp.w.shape}")
    print(f"  dmp_params.npz (weights, c, h, goal)")
    print(f"  pca_model.pkl")

    # ========================================
    # 5. Plot PCA space
    # ========================================
    plt.figure(figsize=(9, 9))

    plt.scatter(X_pca[:, 0], X_pca[:, 1], s=80, c="blue", zorder=5, label="Dataset poses")
    for i, lbl in enumerate(labels):
        plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, lbl, fontsize=7)

    plt.plot(circle_pca[:, 0], circle_pca[:, 1], "b--", linewidth=2, alpha=0.5, label=f"PCA circle (r={radius})")
    plt.scatter(circle_pca[0, 0], circle_pca[0, 1], s=120, c="blue", marker="s", zorder=6, label="Circle start")

    plt.plot(dmp_latent[:, 0], dmp_latent[:, 1], "r-", linewidth=2, label="DMP rollout")
    plt.scatter(dmp_latent[0, 0], dmp_latent[0, 1], s=120, c="red", marker="x", zorder=6, label="DMP start")

    plt.xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    plt.ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    plt.title("PCA Space: Dataset + Circle + DMP")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pca_space_visualization.png", dpi=150)
    print("\nSaved pca_space_visualization.png")
    plt.show()

    # ========================================
    # 6. Simulate in MuJoCo
    # ========================================
    print("\nSimulating DMP trajectory in MuJoCo...")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]

    qpos_idx = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in joint_names
    ])

    qvel_idx = np.array([
        model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
        for name in joint_names
    ])

    ctrl_lo = model.actuator_ctrlrange[:12, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:12, 1].copy()

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.30
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

    for i in range(12):
        data.qpos[qpos_idx[i]] = dmp_joint[0, i]
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    # Settle
    for _ in range(3000):
        target = dmp_joint[0]
        for j in range(12):
            q = data.qpos[qpos_idx[j]]
            qd = data.qvel[qvel_idx[j]]
            gc = data.qfrc_bias[qvel_idx[j]]
            tau = kp * (target[j] - q) + kd * (0.0 - qd) + gc
            data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
        mujoco.mj_step(model, data)

    print(f"Settled. Height: {data.qpos[2]:.4f}")

    # Run trajectory
    start_x = float(data.qpos[0])
    trajectory_idx = 0
    step_counter = 0
    loop_count = 0
    max_loops = 5

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and loop_count < max_loops:
            target = dmp_joint[trajectory_idx]

            for j in range(12):
                q = data.qpos[qpos_idx[j]]
                qd = data.qvel[qvel_idx[j]]
                gc = data.qfrc_bias[qvel_idx[j]]
                tau = kp * (target[j] - q) + kd * (0.0 - qd) + gc
                data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])

            mujoco.mj_step(model, data)
            viewer.sync()

            step_counter += 1
            if step_counter >= steps_per_point:
                step_counter = 0
                trajectory_idx += 1

                if trajectory_idx >= len(dmp_joint):
                    trajectory_idx = 0
                    loop_count += 1
                    dist_x = float(data.qpos[0]) - start_x
                    height = float(data.qpos[2])
                    roll, pitch, yaw = quat_to_euler_wxyz(data.qpos[3:7])
                    print(
                        f"Loop {loop_count}: "
                        f"dist_x={dist_x:.4f}, "
                        f"height={height:.4f}, "
                        f"roll={roll:.3f}, "
                        f"pitch={pitch:.3f}, "
                        f"yaw={yaw:.3f}"
                    )

    final_x = float(data.qpos[0]) - start_x
    final_z = float(data.qpos[2])
    print(f"\nFinal: dist_x={final_x:.4f}, height={final_z:.4f}")


if __name__ == "__main__":
    main()