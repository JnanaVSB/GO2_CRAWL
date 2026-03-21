import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append("/home/jnana/ARLTask/Go2")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from dmp.dmp_rhythmic import DMPs_rhythmic
import mujoco

#paths

CSV_PATH = "/home/jnana/ARLTask/Go2_crawl/dataset/new_dataset_v1.csv"                    # dataset csv
XML_PATH = "/home/jnana/ARLTask/Go2_crawl/go2/scene.xml"                    # go2/scene.xml
DMP_PARAMS_PATH = "/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs20.npz"             # dmp_params.npz
INITIAL_WEIGHTS_PATH = "/home/jnana/ARLTask/Go2_crawl/weights and params/initial_dmp_weights_newV1_bfs20.npy"        # initial_dmp_weights.npy
BEST_WEIGHTS_PATH = "/home/jnana/ARLTask/Go2_crawl/best_weights_from_checkpoint_bfs20.npy"           # best weights from checkpoint
TRAINING_LOG_PATH = "/home/jnana/ARLTask/Go2_crawl/results/cmaes_results_newV1_bfs20/training_log.csv"           # training_log.csv
REWARD_HISTORY_PATH = ""         # reward_history.npy (if saved)
SAVE_PATH = "/home/jnana/ARLTask/Go2_crawl/results/optimization_results_bfs15.png"

N_BFS = 15
N_COMPONENTS = 2
SIM_STEPS = 1000
CONTROL_SUBSTEPS = 4
KP = 80.0
KD = 4.0
RADIUS = 0.4


df = pd.read_csv(CSV_PATH)
labels = df.iloc[:, 0].astype(str).values
X = df.iloc[:, 1:].values.astype(np.float64)

pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X)
pca_mean = np.mean(X_pca, axis=0)

dmp_params = np.load(DMP_PARAMS_PATH)
initial_weights = np.load(INITIAL_WEIGHTS_PATH)
best_weights = np.load(BEST_WEIGHTS_PATH)

joint_names_csv = df.columns[1:].tolist()


# Build initial circle trajectory

theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
circle_pca = np.column_stack([
    pca_mean[0] + RADIUS * np.cos(theta),
    pca_mean[1] + RADIUS * np.sin(theta),
])


# Build initial DMP trajectory

dmp_init = DMPs_rhythmic(n_dmps=2, n_bfs=N_BFS, ay=np.ones(2) * 10.0)
dmp_init.w = initial_weights.copy()
dmp_init.c = dmp_params["c"].copy()
dmp_init.h = dmp_params["h"].copy()
dmp_init.goal = dmp_params["goal"].copy()
init_latent, _, _ = dmp_init.rollout()
init_joint = pca.inverse_transform(init_latent)

# Build optimized DMP trajectory

dmp_best = DMPs_rhythmic(n_dmps=2, n_bfs=N_BFS, ay=np.ones(2) * 10.0)
dmp_best.w = best_weights.reshape(N_COMPONENTS, N_BFS).copy()
dmp_best.c = dmp_params["c"].copy()
dmp_best.h = dmp_params["h"].copy()
dmp_best.goal = dmp_params["goal"].copy()
best_latent, _, _ = dmp_best.rollout()
best_joint = pca.inverse_transform(best_latent)


# Run simulation with best weights

model = mujoco.MjModel.from_xml_path(XML_PATH)
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

# Reset and settle
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.30
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
for i in range(12):
    data.qpos[qpos_idx[i]] = best_joint[0, i]
data.qvel[:] = 0
mujoco.mj_forward(model, data)

for _ in range(3000):
    target = best_joint[0]
    for j in range(12):
        q = data.qpos[qpos_idx[j]]
        qd = data.qvel[qvel_idx[j]]
        gc = data.qfrc_bias[qvel_idx[j]]
        tau = KP * (target[j] - q) + KD * (0.0 - qd) + gc
        data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
    mujoco.mj_step(model, data)

# Run trajectory and collect data
start_x = float(data.qpos[0])
start_y = float(data.qpos[1])
height_history = []
x_history = []
y_history = []
velocity_history = []
joint_pos_history = []

for t in range(SIM_STEPS):
    target = best_joint[t % len(best_joint)]
    for _ in range(CONTROL_SUBSTEPS):
        for j in range(12):
            q = data.qpos[qpos_idx[j]]
            qd = data.qvel[qvel_idx[j]]
            gc = data.qfrc_bias[qvel_idx[j]]
            tau = KP * (target[j] - q) + KD * (0.0 - qd) + gc
            data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
        mujoco.mj_step(model, data)

    height_history.append(float(data.qpos[2]))
    x_history.append(float(data.qpos[0]) - start_x)
    y_history.append(float(data.qpos[1]) - start_y)
    vel = np.mean(np.abs(data.qvel[qvel_idx]))
    velocity_history.append(vel)
    joint_pos_history.append(data.qpos[qpos_idx].copy())

height_history = np.array(height_history)
x_history = np.array(x_history)
y_history = np.array(y_history)
velocity_history = np.array(velocity_history)
joint_pos_history = np.array(joint_pos_history)

# Load training log

try:
    log_df = pd.read_csv(TRAINING_LOG_PATH)
    has_log = True
except:
    has_log = False

try:
    reward_history = np.load(REWARD_HISTORY_PATH)
    has_reward_history = True
except:
    has_reward_history = False


# Create figure

fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

# --- Plot 1: PCA Space ---
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(X_pca[:, 0], X_pca[:, 1], s=50, c='dodgerblue', alpha=0.8, label='Dataset poses', zorder=5)
for i, lbl in enumerate(labels):
    ax1.text(X_pca[i, 0] + 0.01, X_pca[i, 1] + 0.01, lbl, fontsize=5, alpha=0.7)
ax1.plot(circle_pca[:, 0], circle_pca[:, 1], '--', color='gray', linewidth=1.5, alpha=0.5, label='Initial Circle')
ax1.plot(best_latent[:, 0], best_latent[:, 1], '-', color='red', linewidth=2, label='Optimized Trajectory')
ax1.scatter(best_latent[0, 0], best_latent[0, 1], s=120, c='green', zorder=6, label='Start')
ax1.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.2f}%)")
ax1.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.2f}%)")
ax1.set_title("Trajectory in PCA Space")
ax1.legend(fontsize=7, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# --- Plot 2: Optimization Progress ---
ax2 = fig.add_subplot(gs[0, 1])
if has_log:
    gens = log_df['gen'].values
    best_r = log_df['best_reward'].values
    avg_r = log_df['avg_reward'].values
    
    ax2.fill_between(gens, avg_r, best_r, alpha=0.15, color='blue')
    ax2.plot(gens, avg_r, '-', color='lightblue', linewidth=1, alpha=0.7, label='Avg Reward')
    ax2.plot(gens, best_r, '-', color='blue', linewidth=2, label='Best Reward')
    
    best_idx = np.argmax(best_r)
    ax2.scatter(gens[best_idx], best_r[best_idx], s=100, c='black', marker='*', zorder=6, label='Best')
    
    ax2.legend(fontsize=8)
ax2.set_xlabel("Generation")
ax2.set_ylabel("Reward")
ax2.set_title("Optimization Progress")
ax2.grid(True, alpha=0.3)


# --- Plot 3: Joint Trajectories ---
ax3 = fig.add_subplot(gs[0, 2])
joint_labels = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
colors = plt.cm.tab20(np.linspace(0, 1, 12))
for j in range(12):
    ax3.plot(best_joint[:, j], color=colors[j], linewidth=1, alpha=0.8, label=joint_labels[j])
ax3.set_xlabel("Trajectory Point")
ax3.set_ylabel("Joint Angle (rad)")
ax3.set_title("Optimized Joint Trajectories")
ax3.legend(fontsize=5, ncol=3, loc='upper right')
ax3.grid(True, alpha=0.3)

# --- Plot 4: Forward Distance over Time ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(x_history, 'b-', linewidth=1.5, label='Forward (x)')
ax4.plot(y_history, 'r-', linewidth=1.5, alpha=0.7, label='Lateral (y)')
ax4.set_xlabel("Simulation Step")
ax4.set_ylabel("Distance (m)")
ax4.set_title("Distance Over Time")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# --- Plot 5: Height Profile ---
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(height_history, 'g-', linewidth=1.5)
ax5.axhline(y=0.13, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Min Height')
ax5.set_xlabel("Simulation Step")
ax5.set_ylabel("Height (m)")
ax5.set_title("Base Height Profile")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# --- Plot 6: Joint Velocity Profile (per-joint) ---
ax6 = fig.add_subplot(gs[1, 2])
joint_vel_history = []
# Re-run simulation to collect per-joint velocities
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.30
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
for i in range(12):
    data.qpos[qpos_idx[i]] = best_joint[0, i]
data.qvel[:] = 0
mujoco.mj_forward(model, data)

for _ in range(3000):
    target = best_joint[0]
    for j in range(12):
        q = data.qpos[qpos_idx[j]]
        qd = data.qvel[qvel_idx[j]]
        gc = data.qfrc_bias[qvel_idx[j]]
        tau = KP * (target[j] - q) + KD * (0.0 - qd) + gc
        data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
    mujoco.mj_step(model, data)

for t in range(SIM_STEPS):
    target = best_joint[t % len(best_joint)]
    for _ in range(CONTROL_SUBSTEPS):
        for j in range(12):
            q = data.qpos[qpos_idx[j]]
            qd = data.qvel[qvel_idx[j]]
            gc = data.qfrc_bias[qvel_idx[j]]
            tau = KP * (target[j] - q) + KD * (0.0 - qd) + gc
            data.ctrl[j] = np.clip(tau, ctrl_lo[j], ctrl_hi[j])
        mujoco.mj_step(model, data)
    joint_vel_history.append(np.abs(data.qvel[qvel_idx]).copy())

joint_vel_history = np.array(joint_vel_history)
total_vel = np.sum(joint_vel_history, axis=1)

ax6.plot(total_vel, color='purple', linewidth=1.5)
ax6.fill_between(range(len(total_vel)), total_vel, alpha=0.2, color='purple')
ax6.set_xlabel("Simulation Step")
ax6.set_ylabel("Joint Velocity Magnitude")
ax6.set_title("Joint Velocity Profile")
ax6.grid(True, alpha=0.3)

# --- Plot 7: XY Path (top-down view) ---
ax7 = fig.add_subplot(gs[2, 0])
ax7.plot(x_history, y_history, 'b-', linewidth=1.5)
ax7.scatter(0, 0, s=100, c='green', zorder=6, label='Start')
ax7.scatter(x_history[-1], y_history[-1], s=100, c='red', zorder=6, label='End')
ax7.set_xlabel("X (forward)")
ax7.set_ylabel("Y (lateral)")
ax7.set_title("Top-Down Path")
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)
ax7.set_aspect('equal')

# --- Plot 8: Actual Joint Angles During Simulation ---
ax8 = fig.add_subplot(gs[2, 1])
for j in range(12):
    ax8.plot(joint_pos_history[:, j], color=colors[j], linewidth=0.8, alpha=0.7)
ax8.set_xlabel("Simulation Step")
ax8.set_ylabel("Joint Angle (rad)")
ax8.set_title("Actual Joint Angles (Simulation)")
ax8.grid(True, alpha=0.3)

# --- Plot 9: Results Summary ---
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')

final_dist_x = x_history[-1]
final_dist_y = y_history[-1]
final_height = height_history[-1]
avg_height = np.mean(height_history)
min_height = np.min(height_history)
max_vel = np.max(velocity_history)
mean_vel = np.mean(velocity_history)

if has_reward_history:
    initial_reward = reward_history[0]
    best_reward = np.max(reward_history)
    total_gens = len(reward_history)
elif has_log:
    initial_reward = log_df['best_reward'].iloc[0]
    best_reward = log_df['overall_best'].max()
    total_gens = len(log_df)
else:
    initial_reward = 0
    best_reward = 0
    total_gens = 0

summary_text = f"""OPTIMIZATION RESULTS
{'='*40}

Best Reward: {best_reward:.4f}
Total Generations: {total_gens}

PD Control Gains:
  KP: {KP:.2f}
  KD: {KD:.2f}

Trajectory Info:
  Number of points: {len(best_joint)}
  Joint dimensions: 12
  PCA components: {N_COMPONENTS}
  DMP basis functions: {N_BFS}
  PCA variance: {sum(pca.explained_variance_ratio_)*100:.1f}%

Simulation Results:
  Forward distance: {final_dist_x:.4f} m
  Lateral distance: {final_dist_y:.4f} m
  Final height: {final_height:.4f} m
  Avg height: {avg_height:.4f} m
  Min height: {min_height:.4f} m
  Max velocity: {max_vel:.4f} rad/s
  Mean velocity: {mean_vel:.4f} rad/s

Reward History:
  Initial: {initial_reward:.4f}
  Best: {best_reward:.4f}
  Improvement: {best_reward - initial_reward:.4f}"""

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle("Go2 Quadruped Locomotion — PCA + DMP + CMA-ES Optimization Results",
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(SAVE_PATH, dpi=200, bbox_inches='tight')
print(f"Saved to {SAVE_PATH}")
plt.show()