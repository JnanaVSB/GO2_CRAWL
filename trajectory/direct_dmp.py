"""
Direct DMP Trajectory Generator.

Converts a flat weight vector into a joint angle trajectory using:
    1. Reshape weights into (n_joints, n_bfs)
    2. Rhythmic DMP rollout directly in 12-dim joint space

No PCA, no dimensionality reduction. Each joint has its own DMP.
Weight vector size = n_joints * n_bfs (e.g. 12 * 10 = 120 params).

Weight initialization trains the DMP to imitate the dataset poses
as a periodic trajectory directly in joint space.
"""

import os
import numpy as np
import pandas as pd

from trajectory.base import TrajectoryGenerator
from dmp.dmp_rhythmic import DMPs_rhythmic


class DirectDMPTrajectory(TrajectoryGenerator):

    def __init__(
        self,
        dmp_params_path,
        n_bfs,
        dmp_timesteps,
        n_joints=12,
        **kwargs,
    ):
        self.n_joints = n_joints
        self.n_bfs = n_bfs
        self.dmp_timesteps = dmp_timesteps
        self._num_params = n_joints * n_bfs

        # Load DMP params (c, h, goal, y0 from initial fitting)
        dmp_params = np.load(dmp_params_path)
        self.base_c = dmp_params["c"]
        self.base_h = dmp_params["h"]
        self.base_goal = dmp_params["goal"]
        self.base_y0 = dmp_params["y0"]

        # Current weights
        self.weights = np.zeros(self._num_params)

    @property
    def num_params(self):
        return self._num_params

    def generate_trajectory(self, weights_flat=None):
        """
        Generate joint angle trajectory from weights.

        DMP rollout happens directly in 12-joint space.
        No PCA inverse transform needed.

        Returns
        -------
        joint_traj : np.ndarray, shape (dmp_timesteps, n_joints)
        """
        if weights_flat is None:
            weights_flat = self.weights

        w = np.asarray(weights_flat, dtype=np.float64).flatten()
        assert w.shape == (self._num_params,), (
            f"Expected {self._num_params} params, got {w.shape[0]}"
        )

        W = w.reshape(self.n_joints, self.n_bfs)

        dmp = DMPs_rhythmic(
            n_dmps=self.n_joints,
            n_bfs=self.n_bfs,
            ay=np.ones(self.n_joints) * 10.0,
        )
        dmp.w = W.copy()
        dmp.c = self.base_c.copy()
        dmp.h = self.base_h.copy()
        dmp.goal = self.base_goal.copy()
        dmp.y0 = self.base_y0.copy()

        try:
            joint_traj, _, _ = dmp.rollout(timesteps=self.dmp_timesteps)
        except TypeError:
            joint_traj, _, _ = dmp.rollout()
            if len(joint_traj) != self.dmp_timesteps:
                idx = np.linspace(0, len(joint_traj) - 1, self.dmp_timesteps)
                resampled = np.zeros((self.dmp_timesteps, self.n_joints))
                for k in range(self.n_joints):
                    resampled[:, k] = np.interp(
                        idx, np.arange(len(joint_traj)), joint_traj[:, k]
                    )
                joint_traj = resampled

        return joint_traj

    @classmethod
    def ensure_weights(cls, config):
        """
        Generate initial direct DMP weights if they don't exist.

        Builds a small circular oscillation around the base pose
        (initial_angles from env config) for each joint. This ensures
        the robot starts in a known-safe position with gentle periodic
        motion that the optimizer can reshape into actual crawling.

        Same philosophy as PCA mode's circle in latent space, but
        applied directly per-joint.
        """
        policy_cfg = config["policy"]
        agent_cfg = config["agent"]
        env_cfg = config["env"]

        initial_weights_path = agent_cfg["initial_weights_path"]
        dmp_params_path = policy_cfg["dmp_params_path"]

        if os.path.exists(initial_weights_path) and os.path.exists(dmp_params_path):
            print(f"Found existing weights: {initial_weights_path}")
            print(f"Found existing DMP params: {dmp_params_path}")
            return

        print("Initial weights not found. Generating from base pose (direct mode)...")

        n_bfs = policy_cfg["n_bfs"]
        base_angles = np.array(env_cfg["initial_angles"], dtype=np.float64)
        n_joints = len(base_angles)

        # Build a small sinusoidal oscillation around the base pose.
        # Each joint oscillates with a small radius (0.05 rad ~ 3 degrees)
        # and a phase offset so the legs move in a coordinated pattern.
        radius = 0.05
        n_points = 240
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Phase offsets: front-right and rear-left move together (phase 0),
        # front-left and rear-right move together (phase pi).
        # This mimics a trotting-like coordination.
        phase_offsets = np.array([
            0.0, 0.0, 0.0,         # FR hip, thigh, calf
            np.pi, np.pi, np.pi,   # FL hip, thigh, calf
            np.pi, np.pi, np.pi,   # RR hip, thigh, calf
            0.0, 0.0, 0.0,         # RL hip, thigh, calf
        ])

        trajectory = np.zeros((n_points, n_joints))
        for j in range(n_joints):
            trajectory[:, j] = base_angles[j] + radius * np.sin(theta + phase_offsets[j])

        print(f"Base pose circle: {n_joints} joints, radius={radius} rad, {n_points} points")

        # Fit the DMP to the smooth circular trajectory
        dmp = DMPs_rhythmic(
            n_dmps=n_joints,
            n_bfs=n_bfs,
            ay=np.ones(n_joints) * 10.0,
        )
        dmp.imitate_path(y_des=trajectory.T)
        print(f"DMP weights shape: {dmp.w.shape}")

        os.makedirs(os.path.dirname(initial_weights_path), exist_ok=True)
        np.save(initial_weights_path, dmp.w)
        print(f"Saved: {initial_weights_path}")

        os.makedirs(os.path.dirname(dmp_params_path), exist_ok=True)
        np.savez(dmp_params_path, weights=dmp.w, c=dmp.c, h=dmp.h,
                 goal=dmp.goal, y0=dmp.y0)
        print(f"Saved: {dmp_params_path}")