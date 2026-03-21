import numpy as np
import mujoco
import pandas as pd
from sklearn.decomposition import PCA
from dmp.dmp_rhythmic import DMPs_rhythmic


class CrawlEvaluator:
    def __init__(
        self,
        xml_path="/home/jnana/ARLTask/Go2/go2/scene.xml",
        csv_path="/home/jnana/ARLTask/Go2/new_dataset_v1.csv",
        dmp_params_path="/home/jnana/ARLTask/Go2/dmp_params_newV1.npz", #change dmp vales here.
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


    # def _compute_step_reward(self):
    #     x_now = float(self.data.qpos[0])

    #     dx = x_now - self.prev_x
    #     self.prev_x = x_now

    #     reward = 0.0
    #     terms = {}

    #     if dx > 0.0003:
    #         reward += 1000.0 * dx
    #     elif dx < -0.0003:
    #         reward -= 500.0 * abs(dx)

    #     joint_vel = self.data.qvel[self.qvel_index]
    #     if hasattr(self, 'prev_joint_vel'):
    #         jerk = np.mean(np.abs(joint_vel - self.prev_joint_vel))
    #         reward -= 0.005 * jerk
    #     self.prev_joint_vel = joint_vel.copy()

    #     terms["dx"] = dx
    #     terms["ctrl_cost"] = 0.0

        return reward, terms

# working well enough but scratching belly.

    # def _compute_step_reward(self):
    #         x_now = float(self.data.qpos[0])
    #         height = self.data.qpos[2]

    #         dx = x_now - self.prev_x
    #         self.prev_x = x_now

    #         roll, pitch, yaw = self.quat_to_euler_wxyz(self.data.qpos[3:7])

    #         reward = 0.0
    #         terms = {}

    #         reward += 500.0 * max(dx, 0)
    #         reward += 100.0 * min(dx, 0)

    #         if height < 0.08:
    #             reward -= 0.1

    #         reward -= 0.01 * abs(pitch)
    #         reward -= 0.005 * abs(roll)

    #         terms["dx"] = dx
    #         terms["ctrl_cost"] = 0.0

    #         return reward, terms

    # def _compute_step_reward(self):
    #         x_now = float(self.data.qpos[0])
    #         height = self.data.qpos[2]

    #         dx = x_now - self.prev_x
    #         self.prev_x = x_now

    #         total_distance = x_now - self.initial_x

    #         roll, pitch, yaw = self.quat_to_euler_wxyz(self.data.qpos[3:7])

    #         reward = 0.0
    #         terms = {}

    #         if dx > 0:
    #             reward += 200.0 * dx
    #             reward += 50.0 * total_distance / self.sim_steps
    #         else:
    #             reward += 50.0 * dx

    #         if height < 0.08:
    #             reward -= 1.0

    #         reward -= 0.3 * abs(pitch)
    #         reward -= 0.1 * abs(roll)

    #         joint_vel = self.data.qvel[self.qvel_index]
    #         if hasattr(self, 'prev_joint_vel'):
    #             jerk = np.mean(np.abs(joint_vel - self.prev_joint_vel))
    #             reward -= 0.1 * jerk
    #         self.prev_joint_vel = joint_vel.copy()

    #         terms["dx"] = dx
    #         terms["total_distance"] = total_distance
    #         terms["ctrl_cost"] = 0.0

            # return reward, terms

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

    # def evaluate_weights(self, weights_flat, reverse=False, verbose=True):
    #     self.reset()
    #     latent_traj, joint_traj = self.build_trajectory_from_weights(weights_flat, reverse=reverse)
    #     loop_closure = np.linalg.norm(latent_traj[-1] - latent_traj[0])

    #     total_reward = 0.0
    #     terminated = False
    #     accum_ctrl_cost = 0.0
    #     steps_completed = 0

    #     for t in range(self.sim_steps):
    #         target = joint_traj[t % len(joint_traj)]

    #         for _ in range(self.control_substeps):
    #             for j in range(12):
    #                 current_pos = self.data.qpos[self.qpos_index[j]]
    #                 current_vel = self.data.qvel[self.qvel_index[j]]
    #                 gravity_comp = self.data.qfrc_bias[self.qvel_index[j]]

    #                 tau = self.kp * (target[j] - current_pos) + self.kd * (0.0 - current_vel) + gravity_comp
    #                 self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

    #             mujoco.mj_step(self.model, self.data)
    #             steps_completed += 1

    #             reward_t, terms = self._compute_step_reward()
    #             total_reward += reward_t
    #             accum_ctrl_cost += terms["ctrl_cost"]

    #             if self._terminated():
    #                 total_reward -= self.fall_penalty
    #                 terminated = True
    #                 break

    #         if terminated:
    #             break

    #     total_reward -= 10.0 * loop_closure

    #     final_x = float(self.data.qpos[0])
    #     final_y = float(self.data.qpos[1])
    #     final_z = float(self.data.qpos[2])
    #     final_roll, final_pitch, final_yaw = self.quat_to_euler_wxyz(self.data.qpos[3:7])

    #     info = {
    #         "reward": float(total_reward),
    #         "distance_x": final_x - self.initial_x,
    #         "distance_y": final_y - self.initial_y,
    #         "final_height": final_z,
    #         "final_roll": final_roll,
    #         "final_pitch": final_pitch,
    #         "final_yaw": final_yaw,
    #         "terminated": terminated,
    #         "trajectory_length": len(joint_traj),
    #         "accum_ctrl_cost": float(accum_ctrl_cost),
    #         "sim_steps_completed": steps_completed,
    #         "latent_traj": latent_traj,
    #         "joint_traj": joint_traj,
    #     }

    #     if verbose:
    #         print(f"Reward: {info['reward']:.6f}")
    #         print(f"distance_x: {info['distance_x']:.6f}")
    #         print(f"distance_y: {info['distance_y']:.6f}")
    #         print(f"final_height: {info['final_height']:.6f}")
    #         print(f"terminated: {info['terminated']}")
    #         print(f"sim_steps_completed: {info['sim_steps_completed']}")

    #     return float(total_reward), info

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



# class CrawlEnv:
#     """
#     Black-box trajectory evaluation environment for ARS / ES / CMA-style optimization.

#     One action vector = one full rollout evaluation.
#     Action = flattened DMP weights of shape (n_components * n_bfs,)
#     """

#     def __init__(
#         self,
#         xml_path="/home/jnana/ARLTask/Go2/go2/scene.xml",
#         csv_path="/home/jnana/ARLTask/Go2/updated_go2_crawl_poses_v2.csv",
#         n_components=2,
#         n_bfs=10,
#         sim_steps=600,
#         control_substeps=5,
#         dmp_timesteps=120,
#     ):
#         self.n_components = n_components
#         self.n_bfs = n_bfs
#         self.sim_steps = sim_steps
#         self.control_substeps = control_substeps
#         self.dmp_timesteps = dmp_timesteps

#         # MuJoCo
#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)

#         # Pose dataset and PCA
#         poses = pd.read_csv(csv_path)
#         self.pose_labels = poses.iloc[:, 0].values
#         X = poses.iloc[:, 1:].values

#         if X.shape[1] != 12:
#             raise ValueError(f"Expected 12 joint columns in CSV, got {X.shape[1]}")

#         self.pca = PCA(n_components=n_components)
#         self.pca.fit(X)

#         # DMP template
#         self.base_dmp = DMPs_rhythmic(
#             n_dmps=n_components,
#             n_bfs=n_bfs,
#             ay=np.ones(n_components) * 10.0,
#         )

#         # Joint names
#         self.joint_names = [
#             "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#             "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#             "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
#             "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
#         ]

#         self.qpos_index = np.array([
#             self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#             for name in self.joint_names
#         ])

#         self.qvel_index = np.array([
#             self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#             for name in self.joint_names
#         ])

#         for i in range(self.model.nu):
#             name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#             print(i, name)

#         # Actuator checks
#         if self.model.nu < 12:
#             raise ValueError(f"Model has only {self.model.nu} actuators, expected at least 12")

#         self.ctrl_lo = self.model.actuator_ctrlrange[:12, 0].copy()
#         self.ctrl_hi = self.model.actuator_ctrlrange[:12, 1].copy()

#         # PD gains
#         self.kp = 40.0
#         self.kd = 2.0

#         # Initial stable-ish crawl pose
#         self.initial_angles = np.array([
#             0.0, 1.44, -2.46,
#             0.0, 1.44, -2.46,
#             0.0, 1.48, -2.47,
#             0.0, 1.48, -2.47,
#         ], dtype=float)

#         self.initial_base_height = 0.30

#         # Reward weights
#         self.w_forward = 200.0
#         self.w_lateral = 20.0
#         self.w_yaw = 2.0
#         self.w_tilt = 5.0
#         self.w_height_penalty = 5.0
#         self.w_ctrl = 0.0005
#         self.alive_bonus = 0.02
#         self.fall_penalty = 25.0

#         # Safety thresholds
#         self.min_height = 0.10
#         self.max_abs_roll = np.deg2rad(50)
#         self.max_abs_pitch = np.deg2rad(50)

#         # Action dimensionality
#         self.num_params = self.n_components * self.n_bfs

#         # Runtime state
#         self.initial_x = 0.0
#         self.initial_y = 0.0
#         self.prev_x = 0.0
#         self.prev_y = 0.0

#         print(f"PCA components: {self.n_components}")
#         print(f"DMP basis functions: {self.n_bfs}")
#         print(f"Action dimension: {self.num_params}")
#         print(f"Sim steps per rollout: {self.sim_steps}")
#         print(f"Joint qpos index: {self.qpos_index}")
#         print(f"Joint qvel index: {self.qvel_index}")
#         # print("pca traj min:", pca_traj.min(axis=0))
#         # print("pca traj max:", pca_traj.max(axis=0))
#         # print("pca traj first rows:", pca_traj[:5])

#     def reset(self):
#         mujoco.mj_resetData(self.model, self.data)

#         # Base pose
#         self.data.qpos[0] = 0.0
#         self.data.qpos[1] = 0.0
#         self.data.qpos[2] = self.initial_base_height
#         self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

#         # Joint pose
#         for i in range(12):
#             self.data.qpos[self.qpos_index[i]] = self.initial_angles[i]

#         self.data.qvel[:] = 0.0
#         self.data.ctrl[:] = 0.0
#         mujoco.mj_forward(self.model, self.data)

#         # Settle briefly into the initial pose
#         self._settle_pose(self.initial_angles, steps=150)

#         self.initial_x = float(self.data.qpos[0])
#         self.initial_y = float(self.data.qpos[1])
#         self.prev_x = self.initial_x
#         self.prev_y = self.initial_y

#     def _settle_pose(self, target, steps=150):
#         for _ in range(steps):
#             for j in range(12):
#                 qpos_j = self.data.qpos[self.qpos_index[j]]
#                 qvel_j = self.data.qvel[self.qvel_index[j]]
#                 gravity_comp = self.data.qfrc_bias[self.qvel_index[j]]

#                 tau = (
#                     self.kp * (target[j] - qpos_j)
#                     + self.kd * (0.0 - qvel_j)
#                     + gravity_comp
#                 )
#                 self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

#             mujoco.mj_step(self.model, self.data)


#     def _build_trajectory(self, weights_flat):
#         weights = np.asarray(weights_flat, dtype=float).reshape(self.n_components, self.n_bfs)

#         dmp = DMPs_rhythmic(
#             n_dmps=self.n_components,
#             n_bfs=self.n_bfs,
#             ay=np.ones(self.n_components) * 10.0,
#         )

#         # Start from a meaningful DMP parameterization
#         dmp.c = self.base_dmp.c.copy()
#         dmp.h = self.base_dmp.h.copy()
#         dmp.goal = self.base_dmp.goal.copy()
#         dmp.w = weights.copy()

#         try:
#             pca_trajectory, _, _ = dmp.rollout(timesteps=self.dmp_timesteps)
#         except TypeError:
#             pca_trajectory, _, _ = dmp.rollout()

#         joint_trajectory = self.pca.inverse_transform(pca_trajectory)

#         # If the forward direction is wrong for your model, keep this.
#         joint_trajectory = joint_trajectory[::-1]

#         return joint_trajectory

#     # def _build_trajectory(self, weights_flat):
#     #     weights_flat = np.asarray(weights_flat, dtype=float)
#     #     if weights_flat.shape != (self.num_params,):
#     #         raise ValueError(
#     #             f"Expected action shape {(self.num_params,)}, got {weights_flat.shape}"
#     #         )

#     #     weights = weights_flat.reshape(self.n_components, self.n_bfs)

#     #     dmp = DMPs_rhythmic(
#     #         n_dmps=self.n_components,
#     #         n_bfs=self.n_bfs,
#     #         ay=np.ones(self.n_components) * 10.0,
#     #     )

#     #     dmp.w = weights.copy()
#     #     dmp.c = self.base_dmp.c.copy()
#     #     dmp.h = self.base_dmp.h.copy()
#     #     dmp.goal = self.base_dmp.goal.copy()

#     #     pca_traj, _, _ = dmp.rollout(timesteps=self.dmp_timesteps)
#     #     joint_traj = self.pca.inverse_transform(pca_traj)

#     #     print("pca_traj shape:", pca_traj.shape)
#     #     print("pca_traj min:", np.min(pca_traj, axis=0))
#     #     print("pca_traj max:", np.max(pca_traj, axis=0))
#     #     print("first 5 pca points:\n", pca_traj[:5])

#     #     return joint_traj

#     @staticmethod
#     def quat_to_euler_xyzw_safe(q):
#         """
#         MuJoCo free joint quaternion is stored as [w, x, y, z].
#         Returns roll, pitch, yaw.
#         """
#         w, x, y, z = q

#         # roll (x-axis rotation)
#         sinr_cosp = 2.0 * (w * x + y * z)
#         cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
#         roll = np.arctan2(sinr_cosp, cosr_cosp)

#         # pitch (y-axis rotation)
#         sinp = 2.0 * (w * y - z * x)
#         if abs(sinp) >= 1:
#             pitch = np.sign(sinp) * (np.pi / 2.0)
#         else:
#             pitch = np.arcsin(sinp)

#         # yaw (z-axis rotation)
#         siny_cosp = 2.0 * (w * z + x * y)
#         cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
#         yaw = np.arctan2(siny_cosp, cosy_cosp)

#         return roll, pitch, yaw

#     def _compute_step_reward(self):
#         x_now = float(self.data.qpos[0])
#         y_now = float(self.data.qpos[1])
#         z_now = float(self.data.qpos[2])

#         dx = x_now - self.prev_x
#         dy = y_now - self.prev_y

#         self.prev_x = x_now
#         self.prev_y = y_now

#         quat = self.data.qpos[3:7]
#         roll, pitch, yaw = self.quat_to_euler_xyzw_safe(quat)

#         ctrl_cost = float(np.sum(np.square(self.data.ctrl[:12])))

#         reward = 0.0
#         reward += self.w_forward * dx
#         reward -= self.w_lateral * abs(dy)
#         reward -= self.w_yaw * abs(yaw)
#         reward -= self.w_tilt * (abs(roll) + abs(pitch))
#         reward -= self.w_ctrl * ctrl_cost
#         reward += self.alive_bonus

#         if z_now < 0.15:
#             reward -= self.w_height_penalty * (0.15 - z_now)

#         return reward, {
#             "dx": dx,
#             "dy": dy,
#             "height": z_now,
#             "roll": roll,
#             "pitch": pitch,
#             "yaw": yaw,
#             "ctrl_cost": ctrl_cost,
#         }

#     def _terminated(self):
#         z_now = float(self.data.qpos[2])
#         quat = self.data.qpos[3:7]
#         roll, pitch, _ = self.quat_to_euler_xyzw_safe(quat)

#         if z_now < self.min_height:
#             return True
#         if abs(roll) > self.max_abs_roll:
#             return True
#         if abs(pitch) > self.max_abs_pitch:
#             return True
#         if not np.isfinite(self.data.qpos).all():
#             return True
#         if not np.isfinite(self.data.qvel).all():
#             return True

#         return False

#     def evaluate(self, weights_flat, render=False, viewer=None):
#         """
#         Evaluate one DMP parameter vector over a full rollout.
#         """
#         self.reset()
#         trajectory = self._build_trajectory(weights_flat)

#         total_reward = 0.0
#         terminated = False
#         reward_terms_accum = {
#             "dx": 0.0,
#             "dy": 0.0,
#             "ctrl_cost": 0.0,
#         }

#         for t in range(self.sim_steps):
#             reversed_trajectory = trajectory[::-1]
#             target = reversed_trajectory[t % len(reversed_trajectory)]

#             # Hold each trajectory point for a few sim steps
#             for _ in range(self.control_substeps):
#                 for j in range(12):
#                     current_pos = self.data.qpos[self.qpos_index[j]]
#                     current_vel = self.data.qvel[self.qvel_index[j]]
#                     gravity_comp = self.data.qfrc_bias[self.qvel_index[j]]

#                     tau = (
#                         self.kp * (target[j] - current_pos)
#                         + self.kd * (0.0 - current_vel)
#                         + gravity_comp
#                     )
#                     self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

#                 mujoco.mj_step(self.model, self.data)

#                 if render and viewer is not None:
#                     viewer.sync()

#                 reward_t, terms = self._compute_step_reward()
#                 total_reward += reward_t
#                 reward_terms_accum["dx"] += terms["dx"]
#                 reward_terms_accum["dy"] += terms["dy"]
#                 reward_terms_accum["ctrl_cost"] += terms["ctrl_cost"]

#                 if self._terminated():
#                     total_reward -= self.fall_penalty
#                     terminated = True
#                     break

#             if terminated:
#                 break

#         final_x = float(self.data.qpos[0])
#         final_y = float(self.data.qpos[1])
#         final_z = float(self.data.qpos[2])

#         quat = self.data.qpos[3:7]
#         roll, pitch, yaw = self.quat_to_euler_xyzw_safe(quat)

#         info = {
#             "reward": float(total_reward),
#             "distance_x": final_x - self.initial_x,
#             "distance_y": final_y - self.initial_y,
#             "final_height": final_z,
#             "final_roll": roll,
#             "final_pitch": pitch,
#             "final_yaw": yaw,
#             "terminated": terminated,
#             "trajectory_length": len(trajectory),
#             "accum_dx": reward_terms_accum["dx"],
#             "accum_dy": reward_terms_accum["dy"],
#             "accum_ctrl_cost": reward_terms_accum["ctrl_cost"],
#         }

#         return float(total_reward), info

# if __name__ == "__main__":
#     env = CrawlEnv(
#         xml_path="/home/jnana/ARLTask/Go2/go2/scene.xml",
#         csv_path="/home/jnana/ARLTask/Go2/updated_go2_crawl_poses_v2.csv",
#         n_components=2,
#         n_bfs=10,
#         sim_steps=400,
#         control_substeps=4,
#         dmp_timesteps=120,
#     )



#     weights = np.zeros(env.num_params, dtype=float)
#     reward, info = env.evaluate(weights)

#     # for i in range(self.model.nu):
#     #     name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
#     #     print(i, name)

#     print("Reward:", reward)
#     for k, v in info.items():
#         print(f"{k}: {v}")

################################################################################################################################3

# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import mujoco
# import pandas as pd
# from sklearn.decomposition import PCA
# from dmp.dmp_rhythmic import DMPs_rhythmic

# class CrawlEnv(gym.Env):
#     def __init__(self, xml_path = "/home/jnana/ARLTask/Go2/go2/scene.xml", csv_path = "/home/jnana/ARLTask/Go2/go2_crawl_poses.csv", n_components=2, n_bfs = 10, max_episode_steps = 1000):
#         super().__init__()

#         self.max_episode_steps = max_episode_steps
#         self.n_components = n_components
#         self.n_bfs = n_bfs

#         self.model = mujoco.MjModel.from_xml_path(xml_path)
#         self.data = mujoco.MjData(self.model)

#         poses = pd.read_csv(csv_path)
#         X = poses.iloc[:,1:].values
#         self.pca = PCA(n_components = n_components)
#         self.pca.fit(X)
#         X_pca = self.pca.transform(X)
#         self.pca_mean = np.mean(X_pca, axis = 0)

#         self.base_dmp = DMPs_rhythmic(n_dmps=n_components, n_bfs=n_bfs, ay = np.ones(n_components)* 10.0,)

#         self.kp = 40 # how fast we are pulling.
#         self.kd = 2 # how much damping is slowing them down.

#         self.joint_name = [
#             "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
#             "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
#             "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
#             "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
#         ]

#         self.qpos_index = np.array([
#             self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#             for name in self.joint_name
#         ])
        
#         self.qvel_index = np.array([
#             self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
#             for name in self.joint_name
#         ])

#         self.ctrl_lo = self.model.actuator_ctrlrange[:,0]
#         self.ctrl_hi = self.model.actuator_ctrlrange[:,1]

#         n_actions = n_components* n_bfs
#         self.action_space = spaces.Box(low = -1.0, high = 1.0, shape=(n_actions,), dtype =np.float32)

#         n_obs = 12 + 12 + 3 + 4 # joint angles + vel + xyz + quaterinians
#         self.observation_space = spaces.Box(low= -np.inf, high = np.inf, shape = (n_obs,), dtype= np.float32)

#         print("qpos_idx:", self.qpos_index)
#         print("qvel_idx:", self.qvel_index)

#     def reset(self, seed =None, options = None):
#         super().reset(seed = seed)

#         mujoco.mj_resetData(self.model, self.data)

#         self.data.qpos[2] = 0.445

#         initial_angles = np.array([
#                 0.0, 1.44, -2.46,
#                 0.0, 1.44, -2.46,
#                 0.0, 1.48, -2.47,
#                 0.0, 1.48, -2.47,
#         ])

#         for i in range(12):
#             self.data.qpos[self.qpos_index[i]] = initial_angles[i]

#         self.data.qvel[:] = 0


#         mujoco.mj_forward(self.model, self.data)

#         self.step_count = 0
#         self.initial_x = self.data.qpos[0]
#         self.trajectory = None
#         self.trajectory_idx = 0

#         obs = self._get_obs()
#         return obs, {}

#     def _get_obs(self):
#         obs = np.concatenate([
#             self.data.qpos[self.qpos_index],
#             self.data.qvel[self.qvel_index],
#             self.data.qpos[0:3],
#             self.data.qpos[3:7],
#         ])
#         return obs.astype(np.float32)

#     def step(self, action):
#         weights = action.reshape(self.n_components, self.n_bfs)
#         dmp = DMPs_rhythmic(
#             n_dmps=self.n_components,
#             n_bfs=self.n_bfs,
#             ay=np.ones(self.n_components) * 10.0,
#         )
#         dmp.w = weights.copy()
#         dmp.c = self.base_dmp.c.copy()
#         dmp.h = self.base_dmp.h.copy()
#         dmp.goal = self.base_dmp.goal.copy()

#         pca_trajectory, _, _ = dmp.rollout()
#         trajectory = self.pca.inverse_transform(pca_trajectory)

#         total_reward = 0.0
#         terminated = False

#         for step_idx in range(self.max_episode_steps):
#             target = trajectory[step_idx % len(trajectory)]

#             for j in range(12):
#                 current_pos = self.data.qpos[self.qpos_index[j]]
#                 current_vel = self.data.qvel[self.qvel_index[j]]
#                 gravity_comp = self.data.qfrc_bias[self.qvel_index[j]]
#                 tau = self.kp * (target[j] - current_pos) + self.kd * (0.0 - current_vel) + gravity_comp
#                 self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

#             mujoco.mj_step(self.model, self.data)
#             self.step_count += 1

#             reward = self._compute_reward()
#             total_reward += reward

#             if self.data.qpos[2] < 0.1:
#                 terminated = True
#                 break

#         truncated = not terminated

#         obs = self._get_obs()
#         info = {
#             "total_distance": self.data.qpos[0] - self.initial_x,
#             "steps": self.step_count,
#         }

#         return obs, total_reward, terminated, truncated, info

#     # def step(self, action):
#     #     if self.trajectory is None:
#     #         try:
#     #             weights = action.reshape(self.n_components, self.n_bfs)
#     #             dmp = DMPs_rhythmic(
#     #                 n_dmps=self.n_components,
#     #                 n_bfs=self.n_bfs,
#     #                 ay=np.ones(self.n_components) * 10.0,
#     #             )
#     #             dmp.w = weights.copy()
#     #             dmp.c = self.base_dmp.c.copy()
#     #             dmp.h = self.base_dmp.h.copy()
#     #             dmp.goal = self.base_dmp.goal.copy()

#     #             pca_trajectory, _, _ = dmp.rollout()
#     #             self.trajectory = self.pca.inverse_transform(pca_trajectory)
#     #             print(f"Trajectory generated: {self.trajectory.shape}")
#     #         except Exception as e:
#     #             print(f"Error generating trajectory: {e}")
#     #             import traceback
#     #             traceback.print_exc()
            
#     #     target = self.trajectory[self.trajectory_idx]

#     #     for j in range(12):
#     #         current_pos = self.data.qpos[self.qpos_index[j]]
#     #         current_vel = self.data.qvel[self.qvel_index[j]]
#     #         gravity = self.data.qfrc_bias[self.qvel_index[j]]
#     #         tau = self.kp * (target[j] - current_pos) + self.kd * (0.0 - current_vel) + gravity
#     #         self.data.ctrl[j] = np.clip(tau, self.ctrl_lo[j], self.ctrl_hi[j])

#     #     mujoco.mj_step(self.model, self.data)
#     #     self.trajectory_idx = (self.trajectory_idx + 1) % len(self.trajectory)
#     #     self.step_count += 1

#     #     reward = self._compute_reward()
#     #     obs = self._get_obs()
#     #     terminated = self.data.qpos[2] < 0.1
#     #     truncated = self.step_count >=  self.max_episode_steps

#     #     info = {}

#     #     if terminated or truncated:
#     #         info["total distance:"] = self.data.qpos[0] - self.initial_x

#     #     return obs, reward, terminated, truncated, info

#     def _compute_reward(self):
#         distance = self.data.qpos[0] - self.initial_x
#         height = self.data.qpos[2]

#         reward = 0.0
#         reward += distance * 1.0
#         if height < 0.15:
#             reward -= 1.0

#         return reward

# if __name__ == "__main__":
#     env = CrawlEnv()
#     obs, info = env.reset()
#     print(f"observation shape = {obs.shape}")
#     print(f"observation : {obs}")

#     action = env.action_space.sample()
#     print(f"\n Action space: {action.shape}")
#     print(f"Action (first  5): {action[:5]}")

#     total_reward = 0
#     done = False

#     while not done:
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
#         done = terminated or truncated

#         print(f"\nEpisode finished!")
#     print(f"Total reward: {total_reward:.4f}")
#     print(f"Steps: {env.step_count}")
#     print(f"Distance: {info.get('total_distance', 0):.4f}")
#     print(f"Final height: {env.data.qpos[2]:.4f}")