import mujoco
import mujoco.viewer
import numpy as np
import csv
import os

xml_path = "go2/scene.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

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

KP = 320.0
KD = 8.0

target_qpos = np.array([
    0.0303, 1.34, -2.15,
    0.0303, 1.34, -2.15,
    0.0365, 1.38, -2.17,
    0.0365, 1.38, -2.17,
])

# target_qpos = np.array([
#     # 0.0905,0.9173,-2.4214,-0.0422,1.2105,-2.5618,0.124,1.1973,-2.3931, -0.0253,1.7193,-2.5166,
#     # -0.0272,1.2131,-2.5345,0.0801,0.9215,-2.4713,-0.0147,1.7131,-2.5009,0.1078,1.3239,-2.4874
#     0.0458, 1.21, -2.53,
#     0.0161, 1.64, -2.46,
#     0.0441, 1.6, -2.42,
#     0.0278, 1.32, -2.49,
# ], dtype=float)

data.qpos[2] = 0.27
for i in range(12):
    data.qpos[qpos_idx[i]] = target_qpos[i]
data.qvel[:] = 0


def settle_robot(target, steps=10000):
    for step in range(steps):
        for j in range(12):
            current_pos = data.qpos[qpos_idx[j]]
            current_vel = data.qvel[qvel_idx[j]]
            gravity_comp = data.qfrc_bias[qvel_idx[j]]
            tau = KP * (target[j] - current_pos) + KD * (0.0 - current_vel) + gravity_comp
            data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
        mujoco.mj_step(model, data)


mujoco.mj_forward(model, data)
settle_robot(target_qpos)

csv_file = "new_dataset_v1.csv"
pose_count = 0

print("=" * 60)
print("POSE DESIGNER - GRAVITY MODE")
print("=" * 60)
print("Workflow:")
print("  1. Viewer opens with robot standing (PD + gravity)")
print("  2. PAUSE the simulation (spacebar)")
print("  3. Adjust joints using sliders")
print("  4. CLOSE the window")
print("  5. Type a label name in terminal")
print("  6. New window opens with PD holding your new pose")
print("  7. If robot is stable, great! If not, adjust more.")
print("=" * 60)

while True:
    print(f"\nPose {pose_count + 1}")
    print(f"Height: {data.qpos[2]:.4f}")
    print("Open viewer -> Pause -> Adjust -> Close -> Label")

    mujoco.viewer.launch(model, data)

    angles = []
    for i in range(12):
        angles.append(round(data.qpos[qpos_idx[i]], 4))
    print(f"Joint angles: {angles}")

    label = input("Enter label (or 'quit' to stop, 'skip' to retry): ")

    if label == "quit":
        break

    if label == "skip":
        print("Skipping this pose. Settling back to previous target...")
        settle_robot(target_qpos)
        continue

    pose_count += 1
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["label"] + joint_names_csv)
        writer.writerow([label] + angles)

    print(f"Saved pose '{label}'!")

    target_qpos = np.array(angles)
    print("Settling robot into new pose with PD controller...")
    settle_robot(target_qpos)
    print(f"Settled. Height: {data.qpos[2]:.4f}")

print(f"\nDone! Saved {pose_count} poses to {csv_file}")

# import mujoco
# import mujoco.viewer
# import numpy as np
# import csv
# import os

# xml_path = "go2/scene.xml"
# model = mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(model)

# model.opt.gravity[:] = [0, 0, 0]
# data.qpos[2] = 0.20

# joint_names_csv = [
#     "FR_hip", "FR_thigh", "FR_calf",
#     "FL_hip", "FL_thigh", "FL_calf",
#     "RR_hip", "RR_thigh", "RR_calf",
#     "RL_hip", "RL_thigh", "RL_calf",
# ]

# joint_names = [
#     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
#     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
# ]

# qpos_idx = np.array([
#     model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#     for name in joint_names
# ])

# target_qpos = np.array([
#     0.0303, 1.64, -2.48,
#     0.0303, 1.64, -2.48,
#     0.0365, 1.6, -2.41,
#     0.0365, 1.6, -2.41,
# ], dtype=float)

# for i in range(12):
#     data.qpos[qpos_idx[i]] = target_qpos[i]

# mujoco.mj_forward(model, data)

# csv_file = "updated_go2_crawl_poses_v4.csv"
# pose_count = 0

# while True:
#     print(f"\nPose {pose_count + 1}: Pause, adjust joints, then CLOSE the window.")
#     mujoco.viewer.launch(model, data)

#     angles = []
#     for i in range(12):
#         angles.append(round(data.qpos[qpos_idx[i]], 4))
#     print(f"Joint angles: {angles}")

#     label = input("Enter label for this pose (or 'quit' to stop): ")
#     if label == "quit":
#         break

#     pose_count += 1
#     file_exists = os.path.exists(csv_file)
#     with open(csv_file, "a", newline="") as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["label"] + joint_names_csv)
#         writer.writerow([label] + angles)

#     print(f"Saved pose '{label}'!")

# print(f"\nDone! Saved {pose_count} poses to {csv_file}")



# import mujoco
# import mujoco.viewer
# import numpy as np
# import csv
# import os

# xml_path = "go2/scene.xml"
# model = mujoco.MjModel.from_xml_path(xml_path)
# data = mujoco.MjData(model)

# joint_names_csv = [
#     "FR_hip", "FR_thigh", "FR_calf",
#     "FL_hip", "FL_thigh", "FL_calf",
#     "RR_hip", "RR_thigh", "RR_calf",
#     "RL_hip", "RL_thigh", "RL_calf",
# ]

# joint_names = [
#     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
#     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
# ]

# qpos_idx = np.array([
#     model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#     for name in joint_names
# ])

# qvel_idx = np.array([
#     model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#     for name in joint_names
# ])

# ctrl_lo = model.actuator_ctrlrange[:, 0]
# ctrl_hi = model.actuator_ctrlrange[:, 1]

# KP = 40.0
# KD = 1.0

# target_qpos = np.array([
#     0.0303, 1.54, -2.3,
#     0.0303, 1.54, -2.3,
#     0.0365, 1.48, -2.39,
#     0.0365, 1.48, -2.39,
# ])

# data.qpos[2] = 0.27
# for i in range(12):
#     data.qpos[qpos_idx[i]] = target_qpos[i]
# data.qvel[:] = 0
# def settle_robot(model, data, target, qpos_idx, qvel_idx, ctrl_lo, ctrl_hi, steps=2000):
#     for step in range(steps):
#         for j in range(12):
#             current_pos = data.qpos[qpos_idx[j]]
#             current_vel = data.qvel[qvel_idx[j]]
#             gravity_comp = data.qfrc_bias[qvel_idx[j]]
#             tau = KP * (target[j] - current_pos) + KD * (0.0 - current_vel) + gravity_comp
#             data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
#         mujoco.mj_step(model, data)

# csv_file = "updated_go2_crawl_poses_gravity.csv"
# pose_count = 0

# mujoco.mj_forward(model, data)
# settle_robot(model, data, target_qpos, qpos_idx, qvel_idx, ctrl_lo, ctrl_hi)

# print("Robot settled. Starting pose collection.")
# print("Workflow: pause -> adjust joints -> close window -> type label -> repeat")

# while True:
#     print(f"\nPose {pose_count + 1}: Open viewer, PAUSE immediately, adjust joints, then CLOSE.")
#     print(f"Height: {data.qpos[2]:.4f}")

#     mujoco.viewer.launch(model, data)

#     angles = []
#     for i in range(12):
#         angles.append(round(data.qpos[qpos_idx[i]], 4))
#     print(f"Joint angles: {angles}")

#     label = input("Enter label for this pose (or 'quit' to stop): ")
#     if label == "quit":
#         break

#     pose_count += 1
#     file_exists = os.path.exists(csv_file)
#     with open(csv_file, "a", newline="") as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["label"] + joint_names_csv)
#         writer.writerow([label] + angles)

#     print(f"Saved pose '{label}'!")

#     target_qpos = np.array(angles)
#     settle_robot(model, data, target_qpos, qpos_idx, qvel_idx, ctrl_lo, ctrl_hi)
#     print(f"Robot settled into new pose. Height: {data.qpos[2]:.4f}")

# print(f"\nDone! Saved {pose_count} poses to {csv_file}")

