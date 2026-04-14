"""
CMA-ES Agent.

Wraps the cma library's CMAEvolutionStrategy.
The agent only does optimization: ask for candidates, receive rewards, tell results.
Evaluation, logging, checkpointing — all handled by the runner.
"""

import numpy as np
import cma


class CMAESAgent:

    def __init__(
        self,
        initial_weights,
        sigma0,
        popsize,
        max_generations,
        seed=None,
    ):
        self.initial_weights = np.asarray(initial_weights, dtype=np.float64).flatten()
        self.sigma0 = sigma0
        self.popsize = popsize
        self.max_generations = max_generations
        self.seed = seed

        self.best_reward = -np.inf
        self.best_weights = self.initial_weights.copy()

        opts = {
            "popsize": self.popsize,
            "maxiter": self.max_generations,
            "verb_disp": 0,
            "verb_log": 0,
            "tolx": 0,
            "tolfun": 0,
            "tolfunhist": 0,
            "tolstagnation": self.max_generations,
            "tolupsigma": 1e20,
            "tolfacupx": 1e20,
        }
        if seed is not None:
            opts["seed"] = seed

        self.es = cma.CMAEvolutionStrategy(
            self.initial_weights.tolist(),
            self.sigma0,
            opts,
        )

    def ask(self):
        """Get candidate weight vectors for this generation."""
        if self.es.stop():
            self.es.sigma = max(self.es.sigma, 0.01)
        candidates = self.es.ask()
        return [np.array(c) for c in candidates]

    def tell(self, candidates, rewards):
        """
        Report rewards for candidates. CMA-ES minimizes,
        so we negate rewards.
        """
        fitness = [-r for r in rewards]
        self.es.tell(candidates, fitness)

        # Track best
        best_idx = np.argmax(rewards)
        if rewards[best_idx] > self.best_reward:
            self.best_reward = rewards[best_idx]
            self.best_weights = np.array(candidates[best_idx]).copy()

    def get_best(self):
        """Return (best_weights, best_reward)."""
        return self.best_weights.copy(), self.best_reward

    def get_state(self):
        """Return state dict for checkpointing."""
        return {
            "es_mean": np.array(self.es.mean),
            "es_sigma": float(self.es.sigma),
            "best_weights": self.best_weights.copy(),
            "best_reward": float(self.best_reward),
        }

    def load_state(self, state):
        """Resume from checkpoint state."""
        self.best_weights = state["best_weights"].copy()
        self.best_reward = float(state["best_reward"])

        opts = {
            "popsize": self.popsize,
            "maxiter": self.max_generations,
            "verb_disp": 0,
            "verb_log": 0,
            "tolx": 0,
            "tolfun": 0,
            "tolfunhist": 0,
            "tolstagnation": self.max_generations,
            "tolupsigma": 1e20,
            "tolfacupx": 1e20,
        }
        if self.seed is not None:
            opts["seed"] = self.seed

        self.es = cma.CMAEvolutionStrategy(
            state["es_mean"].tolist(),
            state["es_sigma"],
            opts,
        )