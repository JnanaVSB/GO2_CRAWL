"""
SAC-DMP Agent.

Uses Stable-Baselines3's SAC to search over DMP weight vectors
through a proper Gymnasium wrapper environment.

The wrapper env encapsulates one full MuJoCo rollout as a single
Gymnasium step. SB3 owns the training loop — action sampling,
replay buffer, critic updates, entropy tuning are all handled
internally by model.learn().

Key design:
    1. SACSearchEnv: a one-step Gymnasium env where action = normalized
       residual weights in [-1,1]^num_weights, step() runs a full
       MuJoCo rollout, returns scalar reward + terminates.
    2. build_sac_model(): constructs the SB3 SAC model with the right
       hyperparameters for episodic weight optimization.
    3. Action mapping: weights = initial_weights + action * action_scale.
       Zero action = safe prior. Actor learns improvements residually.
    4. Observation deltas are normalized by action_scale so they live
       in [-1, 1] naturally when the actor stays within bounds.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

from evaluation.rollout import evaluate_single, build_eval_args


# ---------------------------------------------------------------------------
# Wrapper env: one MuJoCo rollout = one Gymnasium step
# ---------------------------------------------------------------------------

class SACSearchEnv(gym.Env):
    """
    Gymnasium env for SAC-based DMP weight optimization.

    Each step():
        1. Receives normalized action in [-1, 1]^num_weights
        2. Maps to actual DMP weights: initial + action * scale
        3. Runs a full MuJoCo rollout via evaluate_single()
        4. Returns total reward, terminates immediately

    Observation encodes search history with normalized deltas and
    raw reward stats so the critic has well-scaled inputs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_cls,
        env_kwargs,
        policy_cls,
        policy_kwargs,
        initial_weights,
        sim_steps,
        reward_fn_name,
        termination_fn_name,
        reward_cfg,
        total_episodes,
        action_scale=0.1,
        reward_scale=10.0,
    ):
        super().__init__()

        self.env_cls = env_cls
        self.env_kwargs = env_kwargs
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs
        self.initial_weights = np.asarray(initial_weights, dtype=np.float32).flatten()
        self.num_weights = self.initial_weights.size
        self.action_scale = float(action_scale)
        self.reward_scale = float(reward_scale)

        self.sim_steps = sim_steps
        self.reward_fn_name = reward_fn_name
        self.termination_fn_name = termination_fn_name
        self.reward_cfg = reward_cfg
        self.total_episodes = total_episodes

        # Normalized action space: [-1, 1] for all dimensions
        self.action_space = spaces.Box(
            low=-np.ones(self.num_weights, dtype=np.float32),
            high=np.ones(self.num_weights, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: norm_delta_best + norm_delta_last + reward stats + progress
        # = num_weights + num_weights + 4 = 2 * num_weights + 4
        self.obs_dim = 2 * self.num_weights + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
        )

        # Search state
        self._reset_search_state()

    def _reset_search_state(self):
        """Initialize or reset all search tracking state."""
        self.best_reward = -np.inf
        self.best_weights = self.initial_weights.copy()
        self.last_weights = self.initial_weights.copy()
        self.last_reward = 0.0
        self.reward_history = []
        self.episodes_done = 0

    def _action_to_weights(self, action):
        """Map normalized action [-1,1] to actual DMP weights."""
        action = np.clip(action, -1.0, 1.0)
        return self.initial_weights + action * self.action_scale

    def _build_obs(self):
        """
        Build observation from search history.

        Deltas are normalized by action_scale so they naturally
        stay in [-1, 1] when the actor is within bounds.
        Rewards are kept raw (already normalized in reward.py).
        """
        # Normalize deltas by action_scale
        scale = self.action_scale if self.action_scale > 1e-10 else 1.0
        norm_delta_best = (self.best_weights - self.initial_weights) / scale
        norm_delta_last = (self.last_weights - self.initial_weights) / scale

        if len(self.reward_history) > 0:
            best_r = self.best_reward
            mean_r = np.mean(self.reward_history[-50:])
        else:
            best_r = 0.0
            mean_r = 0.0

        progress = self.episodes_done / max(self.total_episodes, 1)

        obs = np.concatenate([
            norm_delta_best,
            norm_delta_last,
            np.array([best_r, mean_r, self.last_reward, progress]),
        ]).astype(np.float32)

        return np.clip(obs, -10.0, 10.0)

    def get_search_state(self):
        """Return full search state for checkpointing."""
        return {
            "best_reward": float(self.best_reward),
            "best_weights": self.best_weights.copy(),
            "last_weights": self.last_weights.copy(),
            "last_reward": float(self.last_reward),
            "reward_history": np.array(self.reward_history),
            "episodes_done": int(self.episodes_done),
        }

    def load_search_state(self, state):
        """Restore full search state from checkpoint."""
        self.best_reward = float(state["best_reward"])
        self.best_weights = np.asarray(state["best_weights"], dtype=np.float32).flatten()
        self.last_weights = np.asarray(state["last_weights"], dtype=np.float32).flatten()
        self.last_reward = float(state["last_reward"])
        self.reward_history = state["reward_history"].tolist()
        self.episodes_done = int(state["episodes_done"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._build_obs(), {}

    def step(self, action):
        """Run one full MuJoCo rollout with the proposed weights."""
        weights = self._action_to_weights(action)

        eval_args = build_eval_args(
            weights,
            self.env_cls,
            self.env_kwargs,
            self.policy_cls,
            self.policy_kwargs,
            self.sim_steps,
            self.reward_fn_name,
            self.termination_fn_name,
            self.reward_cfg,
        )
        result = evaluate_single(eval_args)

        reward = float(result.total_reward)

        # Update search state
        self.last_weights = weights.copy()
        self.last_reward = reward
        self.reward_history.append(reward)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = weights.copy()

        self.episodes_done += 1

        obs = self._build_obs()

        info = {
            "total_reward": reward,
            "distance_x": float(result.distance_x),
            "terminated_early": result.terminated,
            "steps": result.steps,
            "best_reward": self.best_reward,
            "weights": weights.copy(),
            "best_weights": self.best_weights.copy(),
        }

        return obs, reward * self.reward_scale, True, False, info


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_sac_model(env, agent_cfg):
    """
    Build an SB3 SAC model configured for episodic weight optimization.

    Parameters
    ----------
    env : SACSearchEnv
        The wrapper environment.
    agent_cfg : dict
        Agent config section from YAML.

    Returns
    -------
    model : SAC
        Configured SB3 SAC model.
    """
    num_weights = env.num_weights

    # Target entropy: scale down for high-dim episodic search
    target_entropy = agent_cfg.get("target_entropy", "auto")
    if target_entropy == "auto":
        target_entropy = -num_weights * 0.2

    return SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=agent_cfg.get("learning_rate", 3e-4),
        buffer_size=agent_cfg.get("buffer_size", 20_000),
        learning_starts=agent_cfg.get("learning_starts", 200),
        batch_size=agent_cfg.get("batch_size", 64),
        tau=agent_cfg.get("tau", 0.005),
        gamma=agent_cfg.get("gamma", 0.99),
        train_freq=(1, "step"),
        gradient_steps=agent_cfg.get("gradient_steps", 4),
        ent_coef=agent_cfg.get("ent_coef", "auto"),
        target_entropy=target_entropy,
        policy_kwargs=dict(
            net_arch=agent_cfg.get("net_arch", [256, 256]),
        ),
        seed=agent_cfg.get("seed", None),
        device=agent_cfg.get("device", "auto"),
        verbose=0,
    )