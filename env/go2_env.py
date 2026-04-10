"""
Go2 Crawl Gymnasium Environment.

Pure simulation. Takes joint angle targets, steps MuJoCo, returns raw state.
No reward computation, no termination logic. Those are the caller's job.

    Action:      12-dim joint angle targets
    Observation: joint_pos(12) + joint_vel(12) + base_pos(3) +
                 base_quat(4) + base_linvel(3) + base_angvel(3) = 37
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
NUM_JOINTS = 12


class Go2CrawlEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        xml_path,
        control_substeps,
        kp,
        kd,
        initial_base_height,
        initial_angles,
        settle_steps,
    ):
        super().__init__()

        self.control_substeps = control_substeps
        self.kp = kp
        self.kd = kd
        self.initial_base_height = initial_base_height
        self.initial_angles = np.array(initial_angles, dtype=np.float64)
        self.settle_steps = settle_steps

        # Load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Joint index lookup
        self._qpos_idx = np.array([
            self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in JOINT_NAMES
        ])
        self._qvel_idx = np.array([
            self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            ]
            for n in JOINT_NAMES
        ])

        # Actuator limits
        self._ctrl_lo = self.model.actuator_ctrlrange[:NUM_JOINTS, 0].copy()
        self._ctrl_hi = self.model.actuator_ctrlrange[:NUM_JOINTS, 1].copy()

        # Spaces
        self.action_space = spaces.Box(
            low=self._ctrl_lo,
            high=self._ctrl_hi,
            shape=(NUM_JOINTS,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float64,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.initial_base_height
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        for i in range(NUM_JOINTS):
            self.data.qpos[self._qpos_idx[i]] = self.initial_angles[i]

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(self.settle_steps):
            self._apply_pd(self.initial_angles)
            mujoco.mj_step(self.model, self.data)

        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64).flatten()[:NUM_JOINTS]

        for _ in range(self.control_substeps):
            self._apply_pd(action)
            mujoco.mj_step(self.model, self.data)

        return self._obs(), 0.0, False, False, {}

    def _obs(self):
        return np.concatenate([
            self.data.qpos[self._qpos_idx],
            self.data.qvel[self._qvel_idx],
            self.data.qpos[0:3],
            self.data.qpos[3:7],
            self.data.qvel[0:3],
            self.data.qvel[3:6],
        ])

    def _apply_pd(self, target):
        for j in range(NUM_JOINTS):
            q = self.data.qpos[self._qpos_idx[j]]
            qd = self.data.qvel[self._qvel_idx[j]]
            gc = self.data.qfrc_bias[self._qvel_idx[j]]
            tau = self.kp * (target[j] - q) + self.kd * (0.0 - qd) + gc
            self.data.ctrl[j] = np.clip(tau, self._ctrl_lo[j], self._ctrl_hi[j])