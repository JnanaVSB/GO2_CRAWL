"""
DMP Policy for Go2 Crawl.

Converts a flat weight vector into a joint angle trajectory using:
    1. Reshape weights into (n_components, n_bfs)
    2. Rhythmic DMP rollout in PCA latent space
    3. PCA inverse transform to 12-dim joint space
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from dmp.dmp_rhythmic import DMPs_rhythmic


class DMPPolicy:

    def __init__(
        self,
        csv_path,
        dmp_params_path,
        n_components,
        n_bfs,
        dmp_timesteps,
    ):
        self.n_components = n_components
        self.n_bfs = n_bfs
        self.dmp_timesteps = dmp_timesteps
        self.num_params = n_components * n_bfs

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
        self.weights = np.zeros(self.num_params)

    def get_parameters(self):
        """Return current weight vector (flat)."""
        return self.weights.copy()

    def update_policy(self, weights_flat):
        """Set new weight vector (flat)."""
        weights_flat = np.asarray(weights_flat, dtype=np.float64).flatten()
        assert weights_flat.shape == (self.num_params,), (
            f"Expected {self.num_params} params, got {weights_flat.shape[0]}"
        )
        self.weights = weights_flat.copy()

    def _rollout_latent(self, weights_flat):
        """
        Run DMP rollout and return latent trajectory.
        Shared logic for generate_trajectory and generate_latent_trajectory.
        """
        w = np.asarray(weights_flat, dtype=np.float64).flatten()
        assert w.shape == (self.num_params,), (
            f"Expected {self.num_params} params, got {w.shape[0]}"
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

    def initialize_policy(self):
        """Random initialization (for warmup)."""
        self.weights = np.random.randn(self.num_params) * 0.1