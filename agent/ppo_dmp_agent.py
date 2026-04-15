"""
PPO-DMP Agent.

Uses Stable-Baselines3's PPO to search over DMP weight vectors
through a proper Gymnasium wrapper environment.

Same architecture as SAC agent:
    1. PPOSearchEnv: one-step Gymnasium env where action = normalized
       residual weights in [-1,1]^num_weights, step() runs a full
       MuJoCo rollout, returns scalar reward + terminates.
    2. build_ppo_model(): constructs the SB3 PPO model.
    3. Action mapping: weights = initial_weights + action * action_scale.

Key difference from SAC:
    PPO is on-policy. SB3's PPO collects n_steps transitions, then
    runs n_epochs of gradient updates, then discards the data. This
    is all handled internally by model.learn().
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

from evaluation.rollout import evaluate_single, build_eval_args


# ---------------------------------------------------------------------------
# Wrapper env: one MuJoCo rollout = one Gymnasium step
# ---------------------------------------------------------------------------

class PPOSearchEnv(gym.Env):
    """
    Gymnasium env for PPO-based DMP weight optimization.

    Identical to SACSearchEnv — each step() runs a full MuJoCo rollout
    and terminates immediately. Separate class to allow independent
    tuning of observation design if needed later.
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
        obs_traj_steps=32,
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
        self.obs_traj_steps = obs_traj_steps

        self.sim_steps = sim_steps
        self.reward_fn_name = reward_fn_name
        self.termination_fn_name = termination_fn_name
        self.reward_cfg = reward_cfg
        self.total_episodes = total_episodes

        # Trajectory generator for building observations
        self.traj_gen = policy_cls(**policy_kwargs)

        # Compute initial trajectory for the first observation
        full_traj = self.traj_gen.generate_trajectory(self.initial_weights)
        self.n_joints = full_traj.shape[1]
        self.full_traj_len = full_traj.shape[0]

        # Downsample indices
        self.ds_indices = np.linspace(
            0, self.full_traj_len - 1, self.obs_traj_steps, dtype=int
        )

        # Normalized action space: [-1, 1] for all dimensions
        self.action_space = spaces.Box(
            low=-np.ones(self.num_weights, dtype=np.float32),
            high=np.ones(self.num_weights, dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: last_trajectory + best_trajectory + reward stats
        traj_flat_dim = self.obs_traj_steps * self.n_joints
        self.obs_dim = traj_flat_dim * 2 + 4
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

        init_traj = self.traj_gen.generate_trajectory(self.initial_weights)
        self.best_traj_ds = init_traj[self.ds_indices].astype(np.float32)
        self.last_traj_ds = init_traj[self.ds_indices].astype(np.float32)

    def _action_to_weights(self, action):
        """Map normalized action [-1,1] to actual DMP weights."""
        action = np.clip(action, -1.0, 1.0)
        return self.initial_weights + action * self.action_scale

    def _weights_to_traj_ds(self, weights):
        """Generate trajectory from weights and downsample."""
        full_traj = self.traj_gen.generate_trajectory(weights)
        return full_traj[self.ds_indices].astype(np.float32)

    def _build_obs(self):
        """
        Build observation from trajectory shapes.

        The critic sees actual joint angle trajectories rather than
        abstract weight vectors.
        """
        if len(self.reward_history) > 0:
            best_r = self.best_reward
            mean_r = np.mean(self.reward_history[-50:])
        else:
            best_r = 0.0
            mean_r = 0.0

        progress = self.episodes_done / max(self.total_episodes, 1)

        obs = np.concatenate([
            self.last_traj_ds.flatten(),
            self.best_traj_ds.flatten(),
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

        self.best_traj_ds = self._weights_to_traj_ds(self.best_weights)
        self.last_traj_ds = self._weights_to_traj_ds(self.last_weights)

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

        # Update search state + trajectories
        self.last_weights = weights.copy()
        self.last_reward = reward
        self.last_traj_ds = self._weights_to_traj_ds(weights)
        self.reward_history.append(reward)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = weights.copy()
            self.best_traj_ds = self.last_traj_ds.copy()

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

        # Return scaled reward to SB3, raw reward in info
        return obs, reward * self.reward_scale, True, False, info


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_ppo_model(env, agent_cfg):
    """
    Build an SB3 PPO model configured for episodic weight optimization.

    Parameters
    ----------
    env : PPOSearchEnv
        The wrapper environment.
    agent_cfg : dict
        Agent config section from YAML.

    Returns
    -------
    model : PPO
        Configured SB3 PPO model.
    """
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=agent_cfg.get("learning_rate", 3e-4),
        n_steps=agent_cfg.get("n_steps", 64),
        batch_size=agent_cfg.get("batch_size", 64),
        n_epochs=agent_cfg.get("n_epochs", 10),
        gamma=agent_cfg.get("gamma", 0.99),
        gae_lambda=agent_cfg.get("gae_lambda", 0.95),
        clip_range=agent_cfg.get("clip_range", 0.2),
        ent_coef=agent_cfg.get("ent_coef", 0.01),
        vf_coef=agent_cfg.get("vf_coef", 0.5),
        max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=dict(
            net_arch=agent_cfg.get("net_arch", [256, 256]),
        ),
        seed=agent_cfg.get("seed", None),
        device=agent_cfg.get("device", "cpu"),
        verbose=0,
    )