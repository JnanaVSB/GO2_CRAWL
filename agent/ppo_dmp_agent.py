"""
PPO-DMP Agent.

Uses Stable-Baselines3's PPO to search over DMP weight vectors.
Same goal as SAC-DMP: find the weight vector that produces the
best crawling gait via a one-step MDP.

Key difference from SAC-DMP:
    PPO is on-policy. It cannot reuse old data. The workflow is:
        1. Collect a batch of n_steps episodes
        2. Compute advantages (GAE)
        3. Run n_epochs of gradient updates on that batch
        4. Discard the batch, collect fresh data

    This means the runner must:
        - Get action + value + log_prob together from the policy
        - Store each episode's data in the rollout buffer
        - After n_steps episodes, trigger training

Architecture:
    - SB3's PPO provides the actor-critic network, rollout buffer,
      GAE computation, clipping objective, and gradient updates.
    - A DummyDMPEnv satisfies SB3's constructor (gym spaces).
    - The runner drives the loop and does the actual rollouts.

Runner interface:
    - get_action_value_logprob(obs) : policy forward pass
    - add_to_rollout_buffer(...)    : store one episode
    - compute_returns_and_train()   : GAE + gradient updates
    - save() / load()               : checkpointing
"""

import os
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor


# ---------------------------------------------------------------------------
# Dummy env to satisfy SB3's constructor
# ---------------------------------------------------------------------------

class DummyDMPEnv(gym.Env):
    """
    Minimal gym env that defines the action/obs spaces for PPO.
    Never actually stepped during training — the runner handles rollouts.
    """

    def __init__(self, num_weights, action_low, action_high, initial_weights):
        super().__init__()
        self.initial_weights = np.asarray(initial_weights, dtype=np.float32).flatten()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_weights,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.asarray(action_low, dtype=np.float32),
            high=np.asarray(action_high, dtype=np.float32),
            shape=(num_weights,),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        return self.initial_weights.copy(), {}

    def step(self, action):
        return self.initial_weights.copy(), 0.0, True, False, {}


# ---------------------------------------------------------------------------
# PPO-DMP Agent
# ---------------------------------------------------------------------------

class PPODMPAgent:

    def __init__(
        self,
        num_weights,
        initial_weights,
        total_episodes,
        action_range=2.0,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        net_arch=None,
        seed=None,
        device="auto",
    ):
        """
        Parameters
        ----------
        num_weights : int
            Dimensionality of DMP weight vector (n_components * n_bfs).
        initial_weights : np.ndarray
            Initial DMP weights from the circle trajectory fit.
            Action bounds are centered on these values.
        total_episodes : int
            Total training episodes. Passed to SB3's _setup_learn()
            for learning rate scheduling.
        action_range : float
            Search range: initial_weights ± this value.
        learning_rate : float
            LR for actor and critic.
        n_steps : int
            Number of episodes to collect before each PPO update.
            This is the rollout buffer size. PPO updates after every
            n_steps episodes.
        batch_size : int
            Minibatch size for gradient updates within each epoch.
            Must be <= n_steps.
        n_epochs : int
            Number of passes over the rollout buffer per update.
        gamma : float
            Discount factor.
        gae_lambda : float
            GAE lambda for advantage estimation.
        clip_range : float
            PPO clipping parameter.
        ent_coef : float
            Entropy coefficient for exploration.
        vf_coef : float
            Value function loss coefficient.
        max_grad_norm : float
            Max gradient norm for clipping.
        net_arch : list or None
            Hidden layer sizes for actor/critic. Default [256, 256].
        seed : int or None
            Random seed.
        device : str
            "auto", "cpu", or "cuda".
        """
        self.num_weights = num_weights
        self.initial_weights = np.asarray(initial_weights, dtype=np.float32).flatten()
        self.n_steps = n_steps

        if net_arch is None:
            net_arch = [256, 256]

        # Action bounds centered on initial weights
        action_low = self.initial_weights - action_range
        action_high = self.initial_weights + action_range

        # Dummy env with initial weights as observation space
        self.dummy_env = DummyDMPEnv(
            num_weights=num_weights,
            action_low=action_low,
            action_high=action_high,
            initial_weights=self.initial_weights,
        )

        self.model = PPO(
            policy="MlpPolicy",
            env=self.dummy_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=dict(net_arch=net_arch),
            seed=seed,
            device=device,
            verbose=0,
        )

        # Initialize SB3 internal training state (logger, LR schedule, etc.)
        self.model._setup_learn(
            total_timesteps=total_episodes,
            reset_num_timesteps=False,
        )

        # PPO needs _last_obs set for its internal bookkeeping
        self.model._last_obs = np.array([self.initial_weights], dtype=np.float32)

        # Tracking
        self.best_reward = -np.inf
        self.best_weights = None
        self.episodes_done = 0
        self._buffer_count = 0

        # Observation = initial weights (the trajectory the agent is optimizing)
        self._obs = self.initial_weights.copy()

    # ------------------------------------------------------------------
    # Interface for the runner
    # ------------------------------------------------------------------

    def get_obs(self):
        """Return the observation (initial weights) for this one-step MDP."""
        return self._obs.copy()

    def get_action_value_logprob(self, obs=None):
        """
        Get action, value estimate, and log probability from the policy.

        PPO needs all three at collection time for its objective.

        Parameters
        ----------
        obs : np.ndarray or None
            Observation. None uses the initial weights.

        Returns
        -------
        action : np.ndarray, shape (num_weights,)
            DMP weight vector.
        value : th.Tensor, shape (1,)
            Critic's value estimate.
        log_prob : th.Tensor, shape (1,)
            Log probability of the action under current policy.
        """
        if obs is None:
            obs = self._obs

        obs_tensor = obs_as_tensor(
            np.array([obs]), self.model.device,
        )

        with th.no_grad():
            action, value, log_prob = self.model.policy(obs_tensor)

        # Clip action to bounds
        action_np = action.cpu().numpy().flatten()
        action_np = np.clip(
            action_np,
            self.dummy_env.action_space.low,
            self.dummy_env.action_space.high,
        )

        return action_np, value.flatten(), log_prob.flatten()

    def add_to_rollout_buffer(self, obs, action, reward, done, value, log_prob):
        """
        Add one episode result to PPO's rollout buffer.

        Parameters
        ----------
        obs : np.ndarray
            Observation (dummy obs).
        action : np.ndarray
            DMP weight vector.
        reward : float
            Total rollout reward.
        done : bool
            Always True (one-step episode).
        value : th.Tensor
            Value estimate from get_action_value_logprob().
        log_prob : th.Tensor
            Log prob from get_action_value_logprob().
        """
        self.model.rollout_buffer.add(
            obs=np.array([obs]),
            action=np.array([action]),
            reward=np.array([reward]),
            episode_start=np.array([True]),
            value=value,
            log_prob=log_prob,
        )
        self._buffer_count += 1
        self.episodes_done += 1
        self.model.num_timesteps = self.episodes_done

    def is_buffer_full(self):
        """Check if rollout buffer has n_steps episodes and is ready for training."""
        return self._buffer_count >= self.n_steps

    def compute_returns_and_train(self):
        """
        Compute GAE advantages and run PPO gradient updates.

        Call this after collecting n_steps episodes. The buffer is
        reset internally after training.

        Returns
        -------
        trained : bool
            True if training was performed.
        """
        if not self.is_buffer_full():
            return False

        # Compute value for the last observation (for GAE bootstrap)
        last_obs = obs_as_tensor(
            np.array([self._obs]), self.model.device,
        )
        with th.no_grad():
            last_value = self.model.policy.predict_values(last_obs)

        # Compute returns and advantages
        self.model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_value,
            dones=np.array([True]),
        )

        # Update learning rate schedule
        self.model._update_current_progress_remaining(
            self.episodes_done, self.model._total_timesteps,
        )

        # Run PPO gradient updates (n_epochs over the buffer)
        self.model.train()

        # Reset buffer for next collection phase
        self.model.rollout_buffer.reset()
        self._buffer_count = 0

        return True

    def sample_action(self, obs=None, deterministic=False):
        """
        Simple action sampling (for evaluation, not training).
        For training, use get_action_value_logprob() instead.

        Parameters
        ----------
        obs : np.ndarray or None
            Observation. None uses the initial weights.
        deterministic : bool
            If True, return mean action.

        Returns
        -------
        weights : np.ndarray, shape (num_weights,)
        """
        if obs is None:
            obs = self._obs

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action.flatten()

    # ------------------------------------------------------------------
    # Best tracking
    # ------------------------------------------------------------------

    def update_best(self, weights, reward):
        """Track the best weights found so far."""
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_weights = np.array(weights, dtype=np.float64).copy()

    def get_best(self):
        """Return (best_weights, best_reward)."""
        return (
            self.best_weights.copy() if self.best_weights is not None else None,
            self.best_reward,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def get_state(self):
        """Return state dict for checkpointing compatibility."""
        return {
            "best_weights": self.best_weights,
            "best_reward": float(self.best_reward),
            "episodes_done": self.episodes_done,
        }

    def save(self, path):
        """
        Save full SB3 model + best tracking.

        Creates:
            {path}.zip          — SB3 model (policy + optimizer)
            {path}_meta.npz     — best weights, reward, episode count

        Note: PPO has no replay buffer to save (on-policy, data discarded).
        """
        self.model.save(path)
        np.savez(
            path + "_meta.npz",
            best_weights=self.best_weights if self.best_weights is not None else np.array([]),
            best_reward=float(self.best_reward),
            episodes_done=self.episodes_done,
        )

    def load(self, path):
        self.model = PPO.load(path, env=self.dummy_env)

        # Load meta FIRST so episodes_done is correct for _setup_learn
        meta_path = path + "_meta.npz"
        if os.path.exists(meta_path):
            meta = np.load(meta_path, allow_pickle=True)
            bw = meta["best_weights"]
            self.best_weights = bw if bw.size > 0 else None
            self.best_reward = float(meta["best_reward"])
            self.episodes_done = int(meta["episodes_done"])
            self.model.num_timesteps = self.episodes_done

        # Now _setup_learn gets the correct total_timesteps
        self.model._setup_learn(
            total_timesteps=self.episodes_done,
            reset_num_timesteps=False,
        )
        self.model._last_obs = np.array([self.initial_weights], dtype=np.float32)