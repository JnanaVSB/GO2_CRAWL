"""
Continuous simulation of circle trajectory on Go2 in MuJoCo.
Run with: python circle_trajectory_sim.py
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import mujoco
import mujoco.viewer
import sys
import os
sys.path.append("/home/jnana/ARLTask/Go2")
from dmp.dmp_rhythmic import DMPs_rhythmic

# ========================================
# PATHS - fill these in
# ========================================
CSV_PATH = "/home/jnana/ARLTask/Go2/new_dataset_v1.csv"
XML_PATH = "/home/jnana/ARLTask/Go2/go2/scene.xml"

# ========================================
# Parameters
# ========================================
CIRCLE_RADIUS = 0.4
N_CIRCLE_POINTS = 240
N_BFS = 15
KP = 150.0
KD = 15.0
USE_GRAVITY_COMP = False
REVERSE = True
TIME_PER_POINT = 0.01  # seconds per trajectory point
SETTLE_STEPS = 3000

# ========================================
# Load dataset and fit PCA
# ========================================
print("Loading dataset and fitting PCA...")
df = pd.read_csv(CSV_PATH)
labels = df.iloc[:, 0].astype(str).values
X = df.iloc[:, 1:].values.astype(np.float64)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
pca_mean = np.mean(X_pca, axis=0)

print(f"PCA explained variance: {pca.explained_variance_ratio_}")
print(f"Total: {np.sum(pca.explained_variance_ratio_):.4f}")
print(f"PCA mean: {pca_mean}")

# ========================================
# Generate circle and train DMP
# ========================================
print(f"\nGenerating circle (radius={CIRCLE_RADIUS}, points={N_CIRCLE_POINTS})...")
theta = np.linspace(0, 2 * np.pi, N_CIRCLE_POINTS, endpoint=False)
circle_pca = np.column_stack([
    pca_mean[0] + CIRCLE_RADIUS * np.cos(theta),
    pca_mean[1] + CIRCLE_RADIUS * np.sin(theta),
])

print("Training DMP...")
dmp = DMPs_rhythmic(n_dmps=2, n_bfs=N_BFS, ay=np.ones(2) * 10.0)
dmp.imitate_path(y_des=circle_pca.T)
latent_traj, _, _ = dmp.rollout()

# Project to joint space
joint_traj = pca.inverse_transform(latent_traj)
print(f"DMP rollout shape: {latent_traj.shape}")
print(f"Joint trajectory shape: {joint_traj.shape}")

if REVERSE:
    joint_traj = joint_traj[::-1]
    latent_traj = latent_traj[::-1]
    print("Trajectory REVERSED")

# ========================================
# Load MuJoCo model
# ========================================
print("\nLoading MuJoCo model...")
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Joint mapping
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

ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

# ========================================
# Initialize robot
# ========================================
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.30
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
for i in range(12):
    data.qpos[qpos_idx[i]] = joint_traj[0, i]
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# Settle
print(f"Settling for {SETTLE_STEPS} steps...")
for _ in range(SETTLE_STEPS):
    target = joint_traj[0]
    for j in range(12):
        q = data.qpos[qpos_idx[j]]
        qd = data.qvel[qvel_idx[j]]
        tau = KP * (target[j] - q) + KD * (0.0 - qd)
        if USE_GRAVITY_COMP:
            tau += data.qfrc_bias[qvel_idx[j]]
        data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
    mujoco.mj_step(model, data)

settled_height = float(data.qpos[2])
settled_x = float(data.qpos[0])
print(f"Settled. Height: {settled_height:.4f}, X: {settled_x:.4f}")

# ========================================
# Run simulation with viewer
# ========================================
trajectory_idx = 0
loop_count = 0
steps_per_point = max(1, int(TIME_PER_POINT / model.opt.timestep))
step_in_point = 0

print("=" * 60)
print("Starting simulation")
print(f"Trajectory points: {len(joint_traj)}")
print(f"Steps per point: {steps_per_point}")
print(f"KP={KP}, KD={KD}, gravity_comp={USE_GRAVITY_COMP}")
print(f"Reverse={REVERSE}")
print("Close viewer to exit")
print("=" * 60)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        target = joint_traj[trajectory_idx]

        for j in range(12):
            q = data.qpos[qpos_idx[j]]
            qd = data.qvel[qvel_idx[j]]
            tau = KP * (target[j] - q) + KD * (0.0 - qd)
            if USE_GRAVITY_COMP:
                tau += data.qfrc_bias[qvel_idx[j]]
            data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])

        mujoco.mj_step(model, data)
        viewer.sync()

        step_in_point += 1
        if step_in_point >= steps_per_point:
            step_in_point = 0
            trajectory_idx += 1

            if trajectory_idx >= len(joint_traj):
                trajectory_idx = 0
                loop_count += 1
                dist_x = float(data.qpos[0]) - settled_x
                height = float(data.qpos[2])
                print(f"Loop {loop_count}: dist_x={dist_x:.4f}, height={height:.4f}")

print("\nSimulation ended.")