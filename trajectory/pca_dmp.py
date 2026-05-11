"""
PCA + DMP Trajectory Generator.

Converts a flat weight vector into a joint angle trajectory using:
    1. Reshape weights into (n_components, n_bfs)
    2. Rhythmic DMP rollout in PCA latent space
    3. PCA inverse transform to 12-dim joint space

This is the original trajectory method used in the Go2 crawl project.
Weight initialization creates a circular or elliptical trajectory in
PCA space, centered at the dataset mean. The start point of the
trajectory (DMP at t=0) is the point on the shape nearest to X_pca[0]
(first dataset pose projected into latent space).

The joint-space inverse of that start point is saved as `start_joints`
in the DMP params file so the env can be reset to the exact config
the DMP commands at t=0.
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from trajectory.base import TrajectoryGenerator
from dmp.dmp_rhythmic import DMPs_rhythmic


class PCADMPTrajectory(TrajectoryGenerator):

    def __init__(
        self,
        csv_path,
        dmp_params_path,
        n_components,
        n_bfs,
        dmp_timesteps,
        **kwargs,
    ):
        self.n_components = n_components
        self.n_bfs = n_bfs
        self.dmp_timesteps = dmp_timesteps
        self._num_params = n_components * n_bfs

        # Load dataset and fit PCA
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].astype(str).values
        X = df.iloc[:, 1:].values
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X)

        # Load DMP params (c, h, goal from initial fitting)
        dmp_params = np.load(dmp_params_path)
        self.base_c = dmp_params["c"]
        self.base_h = dmp_params["h"]
        self.base_goal = dmp_params["goal"]

        # Current weights
        self.weights = np.zeros(self._num_params)

    @property
    def num_params(self):
        return self._num_params

    def _rollout_latent(self, weights_flat):
        """
        Run DMP rollout and return latent trajectory.
        Shared logic for generate_trajectory and generate_latent_trajectory.
        """
        w = np.asarray(weights_flat, dtype=np.float64).flatten()
        assert w.shape == (self._num_params,), (
            f"Expected {self._num_params} params, got {w.shape[0]}"
        )

        W = w.reshape(self.n_components, self.n_bfs)

        dmp = DMPs_rhythmic(
            n_dmps=self.n_components,
            n_bfs=self.n_bfs,
            ay=np.ones(self.n_components) * 10.0,
        )
        dmp.w = W.copy()
        dmp.c = self.base_c.copy()
        dmp.h = self.base_h.copy()
        dmp.goal = self.base_goal.copy()

        try:
            latent_traj, _, _ = dmp.rollout(timesteps=self.dmp_timesteps)
        except TypeError:
            latent_traj, _, _ = dmp.rollout()
            if len(latent_traj) != self.dmp_timesteps:
                idx = np.linspace(0, len(latent_traj) - 1, self.dmp_timesteps)
                resampled = np.zeros((self.dmp_timesteps, self.n_components))
                for k in range(self.n_components):
                    resampled[:, k] = np.interp(
                        idx, np.arange(len(latent_traj)), latent_traj[:, k]
                    )
                latent_traj = resampled

        return latent_traj

    def generate_trajectory(self, weights_flat=None):
        """
        Generate joint angle trajectory from weights.

        Returns
        -------
        joint_traj : np.ndarray, shape (dmp_timesteps, 12)
        """
        if weights_flat is None:
            weights_flat = self.weights
        latent_traj = self._rollout_latent(weights_flat)
        joint_traj = self.pca.inverse_transform(latent_traj)
        return joint_traj

    def generate_latent_trajectory(self, weights_flat=None):
        """
        Generate both latent and joint trajectories from weights.

        Returns
        -------
        latent_traj : np.ndarray, shape (dmp_timesteps, n_components)
        joint_traj : np.ndarray, shape (dmp_timesteps, 12)
        """
        if weights_flat is None:
            weights_flat = self.weights
        latent_traj = self._rollout_latent(weights_flat)
        joint_traj = self.pca.inverse_transform(latent_traj)
        return latent_traj, joint_traj

    @classmethod
    def ensure_weights(cls, config):
        """
        Generate initial PCA+DMP weights if they don't exist.

        Builds a circle or ellipse in PCA space (per policy.init_shape),
        centered at the dataset mean, with its start point chosen as
        the point on the shape nearest to X_pca[0]. Trains a rhythmic
        DMP to imitate it. Also saves the joint-space inverse of the
        start point as `start_joints`.
        """
        policy_cfg = config["policy"]
        agent_cfg = config["agent"]

        initial_weights_path = agent_cfg["initial_weights_path"]
        dmp_params_path = policy_cfg["dmp_params_path"]

        if os.path.exists(initial_weights_path) and os.path.exists(dmp_params_path):
            print(f"Found existing weights: {initial_weights_path}")
            print(f"Found existing DMP params: {dmp_params_path}")
            return

        print("Initial weights not found. Generating from dataset...")

        csv_path = policy_cfg["csv_path"]
        n_components = policy_cfg["n_components"]
        n_bfs = policy_cfg["n_bfs"]

        df = pd.read_csv(csv_path)
        X = df.iloc[:, 1:].values.astype(np.float64)
        print(f"Dataset: {X.shape[0]} poses, {X.shape[1]} joints")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        pca_mean = np.mean(X_pca, axis=0)
        print(f"PCA explained variance: {pca.explained_variance_ratio_}")

        init_shape = policy_cfg.get("init_shape", "circle")
        target_latent = X_pca[0]
        n_points = 240

        if init_shape == "circle":
            radius = float(np.linalg.norm(X_pca[0] - pca_mean))
            v = target_latent - pca_mean
            phi0 = float(np.arctan2(v[1], v[0]))
            theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + phi0
            circle_pca = np.column_stack([
                pca_mean[0] + radius * np.cos(theta),
                pca_mean[1] + radius * np.sin(theta),
            ])
            print(f"Init shape: circle, radius={radius:.4f}, "
                  f"start_angle={np.degrees(phi0):.2f} deg")

        elif init_shape == "ellipse":
            a = policy_cfg.get("pca1") if policy_cfg.get("pca1") is not None else 0.4
            b = policy_cfg.get("pca2") if policy_cfg.get("pca2") is not None else 0.1
            theta_dense = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
            pts = np.column_stack([
                pca_mean[0] + a * np.cos(theta_dense),
                pca_mean[1] + b * np.sin(theta_dense),
            ])
            phi0 = float(theta_dense[np.argmin(np.linalg.norm(pts - target_latent, axis=1))])
            theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False) + phi0
            circle_pca = np.column_stack([
                pca_mean[0] + a * np.cos(theta),
                pca_mean[1] + b * np.sin(theta),
            ])
            print(f"Init shape: ellipse, a={a}, b={b}, "
                  f"start_angle={np.degrees(phi0):.2f} deg")

        else:
            raise ValueError(
                f"Unknown init_shape: {init_shape!r}. Use 'circle' or 'ellipse'."
            )

        start_joints = pca.inverse_transform(circle_pca[0].reshape(1, -1))[0]
        print(f"start_joints (env base pose): {start_joints}")

        dmp = DMPs_rhythmic(
            n_dmps=n_components,
            n_bfs=n_bfs,
            ay=np.ones(n_components) * 10.0,
        )
        dmp.imitate_path(y_des=circle_pca.T)
        print(f"DMP weights shape: {dmp.w.shape}")

        os.makedirs(os.path.dirname(initial_weights_path), exist_ok=True)
        np.save(initial_weights_path, dmp.w)
        print(f"Saved: {initial_weights_path}")

        os.makedirs(os.path.dirname(dmp_params_path), exist_ok=True)
        np.savez(
            dmp_params_path,
            weights=dmp.w,
            c=dmp.c,
            h=dmp.h,
            goal=dmp.goal,
            start_joints=start_joints,
        )
        print(f"Saved: {dmp_params_path}")