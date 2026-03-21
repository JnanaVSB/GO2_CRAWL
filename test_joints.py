
import mujoco
import mujoco.viewer
import numpy as np
import csv
import os

xml_path = "go2/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


data.qpos[2] = 0.30

joint_names_csv = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

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



target_qpos = np.array([
    0.0303, 1.54, -2.3,
    0.0303, 1.54, -2.3,
    0.0365, 1.48, -2.39,
    0.0365, 1.48, -2.39,
])

for i in range(12):
    data.qpos[qpos_idx[i]] = target_qpos[i]

KP = 40.0
KD = 5.0

mujoco.mj_forward(model, data)

for step in range(2000):
    for j in range(12):
        current_pos = data.qpos[qpos_idx[j]]
        current_vel = data.qvel[qvel_idx[j]]
        gravity_comp = data.qfrc_bias[qvel_idx[j]]
        tau = KP * (target_qpos[j] - current_pos) + KD * (0.0 - current_vel) + gravity_comp
        data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
    mujoco.mj_step(model, data)

csv_file = "updated_go2_crawl_poses_gravity.csv"
pose_count = 0

while True:
    print(f"\nPose {pose_count + 1}: Pause, adjust joints, then CLOSE the window.")
    print(f"Current height: {data.qpos[2]:.4f}")
    mujoco.viewer.launch(model, data)

    angles = []
    for i in range(12):
        angles.append(round(data.qpos[qpos_idx[i]], 4))
    print(f"Joint angles: {angles}")

    label = input("Enter label for this pose (or 'quit' to stop): ")
    if label == "quit":
        break

    pose_count += 1
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + joint_names_csv)
        writer.writerow([label] + angles)

    print(f"Saved pose '{label}'!")

print(f"\nDone! Saved {pose_count} poses to {csv_file}")