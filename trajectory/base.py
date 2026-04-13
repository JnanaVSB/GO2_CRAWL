"""
Base class for trajectory generators.

Every trajectory method (PCA+DMP, direct DMP, CSV replay, future methods)
inherits from TrajectoryGenerator and implements the required interface.

The runners and evaluation code only ever call:
    - generate_trajectory(weights_flat) -> joint angles
    - num_params -> int (search space size for agents)

Everything else is optional.
"""

import numpy as np
from abc import ABC, abstractmethod


class TrajectoryGenerator(ABC):
    """
    Abstract base class for trajectory generators.

    A trajectory generator converts a flat weight vector into a
    joint angle trajectory that can be fed to the Go2 environment.

    Subclasses must implement:
        - generate_trajectory(weights_flat) -> np.ndarray (timesteps, 12)
        - num_params (property) -> int

    Subclasses may optionally implement:
        - generate_latent_trajectory(weights_flat) -> (latent, joints)
        - ensure_weights(config) -> None (classmethod)
        - initialize_policy() -> None
    """

    @abstractmethod
    def generate_trajectory(self, weights_flat):
        """
        Convert a flat weight vector into joint angle targets.

        Parameters
        ----------
        weights_flat : np.ndarray
            Flat weight vector of length num_params.

        Returns
        -------
        joint_traj : np.ndarray, shape (timesteps, 12)
            Joint angle targets for the 12 Go2 joints.
        """
        pass

    @property
    @abstractmethod
    def num_params(self):
        """
        Total number of parameters in the weight vector.

        Returns
        -------
        int
            Search space dimensionality for the optimizer.
        """
        pass

    # ------------------------------------------------------------------
    # Optional interface — subclasses override only if they need to
    # ------------------------------------------------------------------

    def generate_latent_trajectory(self, weights_flat=None):
        """
        Generate both latent and joint trajectories.

        Only meaningful for methods that have a latent space (e.g. PCA).
        Default returns None so that logging.py's hasattr checks
        skip visualization gracefully.

        Returns
        -------
        latent_traj : np.ndarray or None
        joint_traj : np.ndarray or None
        """
        return None, None

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

    def initialize_policy(self):
        """Random initialization (for warmup)."""
        self.weights = np.random.randn(self.num_params) * 0.1

    @classmethod
    def ensure_weights(cls, config):
        """
        Check if initial weights exist on disk, generate if not.

        Each trajectory type knows how to create its own initial weights.
        Default implementation does nothing — subclasses override as needed.

        Parameters
        ----------
        config : dict
            Full config dict with policy, agent, and other sections.
        """
        pass