import numpy as np
import mujoco
import pandas as pd
from sklearn.decomposition import PCA
from dmp.dmp_rhythmic import DMPs_rhythmic


class CrawlEvaluator:
    def __init__(
        self,
        xml_path="/home/jnana/ARLTask/Go2/go2/scene.xml",
        csv_path="/home/jnana/ARLTask/Go2_crawl/go2/scene.xml,
        dmp_params_path="/home/jnana/ARLTask/Go2_crawl/weights and params/dmp_params_newV1_bfs10.npz", #change dmp values here.
        n_components=2,
        n_bfs=10,
        dmp_timesteps=240,
        sim_steps=1000,
        control_substeps=4,
    ):
        self.n_components = n_components
        self.n_bfs = n_bfs
        self.dmp_timesteps = dmp_timesteps
        self.sim_steps = sim_steps
        self.control_substeps = control_substeps
        self.num_params = n_components * n_bfs

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].astype(str).values
        self.joint_names_csv = df.columns[1:].tolist()
        X = df.iloc[:, 1:].values

        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X)

        dmp_params = np.load(dmp_params_path)
        self.base_dmp_c = dmp_params["c"]
        self.base_dmp_h = dmp_params["h"]
        self.base_dmp_goal = dmp_params["goal"]
        print(f"Loaded DMP params: goal={self.base_dmp_goal}")

        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]

        self.qpos_index = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.joint_names
        ])

        self.qvel_index = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in self.joint_names
        ])

        self.ctrl_lo = self.model.actuator_ctrlrange[:12, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:12, 1].copy()

        self.kp = 80.0
        self.kd = 4.0

        self.initial_angles = np.array([
            0.0303, 1.54, -2.3,
            0.0303, 1.54, -2.3,
            0.0365, 1.48, -2.39,
            0.0365, 1.48, -2.39,
        ], dtype=float)
        self.initial_base_height = 0.30

        self.min_height = 0.14
        self.max_abs_roll = np.deg2rad(50.0)
        self.max_abs_pitch = np.deg2rad(50.0)
        self.fall_penalty = 25.0

        self.initial_x = 0.0
        self.initial_y = 0.0
        self.prev_x = 0.0
        self.prev_y = 0.0

        print("PCA explained variance:", self.pca.explained_variance_ratio_)
        # print("Joint qpos index:", self.qpos_index)
        # print("Joint qvel index:", self.qvel_index)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.initial_base_height
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

        for i in range(12):
            self.data.qpos[self.qpos_index[i]] = self.initial_angles[i]

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._settle_pose(self.initial_angles, steps=1000)

        self.initial_x = float(self.data.qpos[0])
        self.initial_y = float(self.data.qpos[1])
        self.prev_x = self.initial_x
        self.prev_y = self.initial_y
        self.prev_joint_vel = np.zeros(12)

    def _settle_pose(self, target, steps=1000):
        for _ in range(steps):
            for j in range(12):
                q = self.data.qpos[self.qpos_index[j]]
                qd = self.data.qvel[self.qvel_index[j]]
                gravity_comp = self.data.qfrc_bias[self.qvel_index[j]]

                tau = self.kp * (target[j] - q) + self.kd * (0.0 - qd) + gravity_comp
                self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

            mujoco.mj_step(self.model, self.data)

    def build_trajectory_from_weights(self, weights_flat, reverse=False):
        weights_flat = np.asarray(weights_flat, dtype=float)
        if weights_flat.shape != (self.num_params,):
            raise ValueError(f"Expected weights shape {(self.num_params,)}, got {weights_flat.shape}")

        weights = weights_flat.reshape(self.n_components, self.n_bfs)

        dmp = DMPs_rhythmic(
            n_dmps=self.n_components,
            n_bfs=self.n_bfs,
            ay=np.ones(self.n_components) * 10.0,
        )
        dmp.w = weights.copy()
        dmp.c = self.base_dmp_c.copy()
        dmp.h = self.base_dmp_h.copy()
        dmp.goal = self.base_dmp_goal.copy()

        try:
            latent_traj, _, _ = dmp.rollout(timesteps=self.dmp_timesteps)
        except TypeError:
            latent_traj, _, _ = dmp.rollout()

            if len(latent_traj) != self.dmp_timesteps:
                idx = np.linspace(0, len(latent_traj) - 1, self.dmp_timesteps)
                resampled = np.zeros((self.dmp_timesteps, self.n_components))
                for k in range(self.n_components):
                    resampled[:, k] = np.interp(idx, np.arange(len(latent_traj)), latent_traj[:, k])
                latent_traj = resampled


        joint_traj = self.pca.inverse_transform(latent_traj)

        if reverse:
            latent_traj = latent_traj[::-1]
            joint_traj = joint_traj[::-1]

        return latent_traj, joint_traj

    @staticmethod
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

    def _compute_step_reward(self):
        x_now = float(self.data.qpos[0])
        y_now = float(self.data.qpos[1])

        dx = x_now - self.prev_x
        dy = y_now - self.prev_y
        self.prev_x = x_now
        self.prev_y = y_now

        reward = 0.0
        terms = {}

        reward += 1000.0 * dx / self.sim_steps

        reward -= 500.0 * abs(dy) / self.sim_steps

        joint_vel = self.data.qvel[self.qvel_index]
        reward -= 0.0002 * np.mean(np.abs(joint_vel))

        terms["dx"] = dx
        terms["ctrl_cost"] = 0.0

        return reward, terms



    def _terminated(self):
        z_now = float(self.data.qpos[2])
        roll, pitch, _ = self.quat_to_euler_wxyz(self.data.qpos[3:7])

        if z_now < self.min_height:
            return True
        if z_now > 0.20:
            return True
        if abs(roll) > self.max_abs_roll:
            return True
        if abs(pitch) > self.max_abs_pitch:
            return True
        if not np.isfinite(self.data.qpos).all():
            return True
        if not np.isfinite(self.data.qvel).all():
            return True
        return False



    def evaluate_weights(self, weights_flat, reverse=False, verbose=True):
        self.reset()
        latent_traj, joint_traj = self.build_trajectory_from_weights(weights_flat, reverse=reverse)

        pc_mean = np.mean(self.X_pca, axis=0)
        pc_std = np.std(self.X_pca, axis=0)

        if np.any(latent_traj > pc_mean + 5.0 * pc_std) or np.any(latent_traj < pc_mean - 5.0 * pc_std):
            info = {
                "reward": -50.0,
                "distance_x": 0.0,
                "distance_y": 0.0,
                "final_height": 0.0,
                "final_roll": 0.0,
                "final_pitch": 0.0,
                "final_yaw": 0.0,
                "terminated": True,
                "trajectory_length": len(joint_traj),
                "accum_ctrl_cost": 0.0,
                "sim_steps_completed": 0,
                "latent_traj": latent_traj,
                "joint_traj": joint_traj,
            }
            if verbose:
                print("Trajectory outside bounds — rejected")
            return -50.0, info

        loop_closure = np.linalg.norm(latent_traj[-1] - latent_traj[0])

        total_reward = 0.0
        terminated = False
        accum_ctrl_cost = 0.0
        steps_completed = 0

        for t in range(self.sim_steps):
            target = joint_traj[t % len(joint_traj)]

            for _ in range(self.control_substeps):
                for j in range(12):
                    current_pos = self.data.qpos[self.qpos_index[j]]
                    current_vel = self.data.qvel[self.qvel_index[j]]
                    gravity_comp = self.data.qfrc_bias[self.qvel_index[j]]

                    tau = self.kp * (target[j] - current_pos) + self.kd * (0.0 - current_vel) + gravity_comp
                    self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

                mujoco.mj_step(self.model, self.data)
                steps_completed += 1

                reward_t, terms = self._compute_step_reward()
                total_reward += reward_t
                accum_ctrl_cost += terms["ctrl_cost"]

                if self._terminated():
                    total_reward -= self.fall_penalty
                    terminated = True
                    break

            if terminated:
                break

        total_reward -= 10.0 * loop_closure

        final_x = float(self.data.qpos[0])
        final_y = float(self.data.qpos[1])
        final_z = float(self.data.qpos[2])
        final_roll, final_pitch, final_yaw = self.quat_to_euler_wxyz(self.data.qpos[3:7])

        info = {
            "reward": float(total_reward),
            "distance_x": final_x - self.initial_x,
            "distance_y": final_y - self.initial_y,
            "final_height": final_z,
            "final_roll": final_roll,
            "final_pitch": final_pitch,
            "final_yaw": final_yaw,
            "terminated": terminated,
            "trajectory_length": len(joint_traj),
            "accum_ctrl_cost": float(accum_ctrl_cost),
            "sim_steps_completed": steps_completed,
            "latent_traj": latent_traj,
            "joint_traj": joint_traj,
        }

        if verbose:
            print(f"Reward: {info['reward']:.6f}")
            print(f"distance_x: {info['distance_x']:.6f}")
            print(f"distance_y: {info['distance_y']:.6f}")
            print(f"final_height: {info['final_height']:.6f}")
            print(f"terminated: {info['terminated']}")
            print(f"sim_steps_completed: {info['sim_steps_completed']}")

        return float(total_reward), info



