import numpy as np
import mujoco
import mujoco.viewer
import pickle
import sys
sys.path.append("/home/jnana/ARLTask/Go2")

xml_path = "/home/jnana/ARLTask/Go2/go2/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

joint_trajectory = np.load("/home/jnana/ARLTask/Go2/optimize/initial_joint_trajectory_bfs20_v12.npy")
print(f"Trajectory shape: {joint_trajectory.shape}")
print(f"Trajectory min: {joint_trajectory.min(axis=0)}")
print(f"Trajectory max: {joint_trajectory.max(axis=0)}")

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

ctrl_lo = model.actuator_ctrlrange[:, 0]
ctrl_hi = model.actuator_ctrlrange[:, 1]

data.qpos[2] = 0.27
for i in range(12):
    data.qpos[qpos_idx[i]] = joint_trajectory[0][i]
data.qvel[:] = 0

KP = 40.0
KD = 1.0

mujoco.mj_forward(model, data)

for step in range(3000):
    target = joint_trajectory[0]
    for j in range(12):
        current_pos = data.qpos[qpos_idx[j]]
        current_vel = data.qvel[qvel_idx[j]]
        gravity_comp = data.qfrc_bias[qvel_idx[j]]
        tau = KP * (target[j] - current_pos) + KD * (0.0 - current_vel) + gravity_comp
        data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
    mujoco.mj_step(model, data)

print(f"Settled. Height: {data.qpos[2]:.4f}")

trajectory_idx = 0
loop_count = 0
step_count = 0
steps_per_point = 4

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        target = joint_trajectory[trajectory_idx]

        for j in range(12):
            current_pos = data.qpos[qpos_idx[j]]
            current_vel = data.qvel[qvel_idx[j]]
            gravity_comp = data.qfrc_bias[qvel_idx[j]]
            tau = KP * (target[j] - current_pos) + KD * (0.0 - current_vel) + gravity_comp
            data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])

        mujoco.mj_step(model, data)
        viewer.sync()

        step_count += 1
        if step_count >= steps_per_point:
            step_count = 0
            trajectory_idx += 1

            if trajectory_idx >= len(joint_trajectory):
                trajectory_idx = 0
                loop_count += 1
                print(f"Loop {loop_count}, Height: {data.qpos[2]:.4f}, X: {data.qpos[0]:.4f}")