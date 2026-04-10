"""
ProPS Replay Buffer.

Stores (weights, cost, reward, metadata) tuples from rollout evaluations
and formats them as text for the LLM prompt.

This is the ProPS equivalent of SB3's ReplayBuffer or CMA-ES's
population history. The key difference: the buffer's primary output
is a formatted text string that goes directly into the LLM prompt,
not tensors for gradient computation.

Two feedback modes control how much info the LLM sees per entry:
    - "basic":    w[0]: val; w[1]: val; ... f(w): cost
    - "detailed": same + dist_x, terminated, steps, PCA latent range

Used by ProPSDMPAgent. Could also be used by future LLM-based
optimizers that follow the same pattern.
"""

import numpy as np


class WeightHistoryBuffer:
    """
    Stores (weights, cost, reward, metadata) tuples and formats them
    as text for the LLM prompt.

    Parameters
    ----------
    max_size : int
        Maximum number of entries. When exceeded, the oldest entry
        is dropped (FIFO). This bounds the LLM prompt length.
    """

    def __init__(self, max_size=200):
        self.max_size = max_size
        self.entries = []

    def add(self, weights, cost, reward, metadata=None):
        """
        Add one rollout result to the buffer.

        Parameters
        ----------
        weights : np.ndarray
            DMP weight vector that was evaluated.
        cost : float
            Cost value (= -reward). Used when LLM minimizes.
        reward : float
            Raw reward from rollout. Used when LLM maximizes.
        metadata : dict or None
            Extra rollout info. Recognized keys:
                distance_x : float — forward distance traveled
                terminated : bool  — whether robot fell / went unstable
                steps      : int   — simulation steps before termination
                pca_x_range : tuple (min, max) — PCA PC1 trajectory range
                pca_y_range : tuple (min, max) — PCA PC2 trajectory range
        """
        entry = {
            "weights": np.asarray(weights, dtype=np.float64).flatten().copy(),
            "cost": float(cost),
            "reward": float(reward),
            "metadata": metadata or {},
        }
        self.entries.append(entry)

        if len(self.entries) > self.max_size:
            self.entries.pop(0)

    def size(self):
        """Return number of entries in the buffer."""
        return len(self.entries)

    # ------------------------------------------------------------------
    # Text formatting for LLM prompts
    # ------------------------------------------------------------------

    def format_basic(self, num_weights, objective="cost"):
        """
        Format buffer as text — weights + objective value only.

        Example output line:
            w[0]: 1.234; w[1]: -0.567; ... w[19]: 0.891; f(w): -12.3400

        Parameters
        ----------
        num_weights : int
            Number of weight dimensions to include.
        objective : str
            "cost" or "reward" — which value to show as f(w).

        Returns
        -------
        text : str
            One line per entry, newline-separated.
        """
        lines = []
        for entry in self.entries:
            w = entry["weights"]
            parts = []
            for i in range(num_weights):
                parts.append(f"w[{i}]: {w[i]:.5g}")
            obj_val = entry[objective]
            parts.append(f"f(w): {obj_val:.4f}")
            lines.append("; ".join(parts))
        return "\n".join(lines)

    def format_detailed(self, num_weights, objective="cost"):
        """
        Format buffer as text — weights + objective + rollout details.

        Example output line:
            w[0]: 1.234; ...; f(w): -12.34; dist_x: 0.0523;
            terminated: no; steps: 1000; pca_x: [-0.5, 0.8];
            pca_y: [-0.3, 0.6]

        Parameters
        ----------
        num_weights : int
            Number of weight dimensions to include.
        objective : str
            "cost" or "reward" — which value to show as f(w).

        Returns
        -------
        text : str
            One line per entry, newline-separated.
        """
        lines = []
        for entry in self.entries:
            w = entry["weights"]
            meta = entry["metadata"]

            parts = []
            for i in range(num_weights):
                parts.append(f"w[{i}]: {w[i]:.5g}")

            obj_val = entry[objective]
            parts.append(f"f(w): {obj_val:.4f}")

            if "distance_x" in meta:
                parts.append(f"dist_x: {meta['distance_x']:.4f}")
            if "terminated" in meta:
                parts.append(
                    f"terminated: {'yes' if meta['terminated'] else 'no'}"
                )
            if "steps" in meta:
                parts.append(f"steps: {meta['steps']}")
            if "pca_x_range" in meta:
                lo, hi = meta["pca_x_range"]
                parts.append(f"pca_x: [{lo:.3f}, {hi:.3f}]")
            if "pca_y_range" in meta:
                lo, hi = meta["pca_y_range"]
                parts.append(f"pca_y: [{lo:.3f}, {hi:.3f}]")

            lines.append("; ".join(parts))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Best tracking helpers
    # ------------------------------------------------------------------

    def get_best_by_cost(self):
        """
        Return (weights, cost) with the lowest cost.

        Returns
        -------
        weights : np.ndarray or None
        cost : float (np.inf if buffer is empty)
        """
        if not self.entries:
            return None, np.inf
        best = min(self.entries, key=lambda e: e["cost"])
        return best["weights"].copy(), best["cost"]

    def get_best_by_reward(self):
        """
        Return (weights, reward) with the highest reward.

        Returns
        -------
        weights : np.ndarray or None
        reward : float (-np.inf if buffer is empty)
        """
        if not self.entries:
            return None, -np.inf
        best = max(self.entries, key=lambda e: e["reward"])
        return best["weights"].copy(), best["reward"]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save buffer to npz file.

        Parameters
        ----------
        path : str
            File path (without .npz extension — np.savez adds it).
        """
        if not self.entries:
            np.savez(path, empty=True)
            return

        weights = np.array([e["weights"] for e in self.entries])
        costs = np.array([e["cost"] for e in self.entries])
        rewards = np.array([e["reward"] for e in self.entries])
        meta_list = [e["metadata"] for e in self.entries]

        np.savez(
            path,
            weights=weights,
            costs=costs,
            rewards=rewards,
            metadata=np.array(meta_list, dtype=object),
        )

    def load(self, path):
        """
        Load buffer from npz file.

        Parameters
        ----------
        path : str
            File path to the .npz file.
        """
        data = np.load(path, allow_pickle=True)
        if "empty" in data:
            self.entries = []
            return

        weights = data["weights"]
        costs = data["costs"]
        rewards = data["rewards"]
        meta_arr = data["metadata"]

        self.entries = []
        for i in range(len(weights)):
            meta = meta_arr[i] if meta_arr[i] is not None else {}
            # np.load with allow_pickle can return numpy types — convert
            if hasattr(meta, "item"):
                meta = meta.item()
            self.entries.append({
                "weights": weights[i].copy(),
                "cost": float(costs[i]),
                "reward": float(rewards[i]),
                "metadata": dict(meta) if meta else {},
            })