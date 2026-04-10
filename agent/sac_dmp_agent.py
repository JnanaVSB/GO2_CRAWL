"""
SAC-DMP Agent.

Uses Stable-Baselines3's SAC to search over DMP weight vectors.
One set of weights = one rhythmic gait. The agent's job is to find
the weight vector that produces the best crawling gait.

This is a one-step MDP by design (not a workaround):
    - DMP weights define a fixed cyclic attractor
    - Changing weights mid-episode would break the rhythmic pattern
    - So: one action (weights) per episode, one scalar reward back

Architecture:
    - SB3's SAC provides the policy network, replay buffer, critic,
      entropy tuning, and gradient updates — all battle-tested.
    - A lightweight DummyDMPEnv satisfies SB3's constructor (it needs
      gym spaces), but is never actually stepped during training.
    - The runner drives the loop: asks for weights, does the rollout
      with Go2CrawlEnv + DMPPolicy, and feeds results back here.

Runner interface:
    - sample_action()    : get a weight vector (with SAC exploration)
    - store_transition() : push experience to SB3's replay buffer
    - update()           : trigger SB3's gradient steps
    - save() / load()    : checkpointing
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
import torch


# ---------------------------------------------------------------------------
# Dummy env to satisfy SB3's constructor
# ---------------------------------------------------------------------------

class DummyDMPEnv(gym.Env):
    """
    Minimal gym env that defines the action/obs spaces for SAC.
    Never actually stepped during training — the runner handles rollouts.

    Action space: DMP weight vector (num_weights,)
    Obs space:    initial DMP weights (num_weights,) — the agent sees
                  the starting trajectory it's optimizing around.
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
# SAC-DMP Agent
# ---------------------------------------------------------------------------

class SACDMPAgent:

    def __init__(
        self,
        num_weights,
        initial_weights,
        total_episodes,
        action_range=2.0,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=10_000,
        learning_starts=100,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        target_entropy="auto",
        gradient_steps=1,
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
            How far SAC can explore from the initial weights.
            Action space becomes [initial_weights - range, initial_weights + range].
        learning_rate : float
            LR for actor, critic, and entropy coefficient.
        batch_size : int
            Minibatch size for gradient updates.
        buffer_size : int
            Replay buffer capacity (number of episodes stored).
        learning_starts : int
            Episodes of random exploration before gradient updates begin.
        tau : float
            Soft update coefficient for target critic networks.
        gamma : float
            Discount factor. Minimal effect in one-step MDP but kept
            for API correctness.
        ent_coef : str or float
            Entropy coefficient. "auto" learns it automatically.
        target_entropy : str or float
            Target entropy for auto tuning. "auto" sets -dim(action).
        gradient_steps : int
            Number of gradient updates per episode.
        net_arch : list or None
            Hidden layer sizes for actor/critic. Default [256, 256].
        seed : int or None
            Random seed.
        device : str
            "auto", "cpu", or "cuda".
        """
        self.num_weights = num_weights
        self.initial_weights = np.asarray(initial_weights, dtype=np.float32).flatten()
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size

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

        self.model = SAC(
            policy="MlpPolicy",
            env=self.dummy_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_starts=0,  # We handle this ourselves in the runner
            tau=tau,
            gamma=gamma,
            ent_coef=ent_coef,
            target_entropy=target_entropy,
            gradient_steps=gradient_steps,
            policy_kwargs=dict(net_arch=net_arch),
            seed=seed,
            device=device,
            verbose=0,
        )

        # Initialize SB3's internal training state (logger, learning rate
        # schedule, etc.) so that model.train() can be called directly.
        # Without this, train() crashes because _setup_learn() was never
        # called — that normally happens inside learn().
        self.model._setup_learn(
            total_timesteps=total_episodes,
            reset_num_timesteps=False,
        )

        # Tracking
        self.best_reward = -np.inf
        self.best_weights = None
        self.episodes_done = 0

        # Observation = initial weights (the trajectory the agent is optimizing)
        self._obs = self.initial_weights.copy()

    # ------------------------------------------------------------------
    # Interface for the runner
    # ------------------------------------------------------------------

    def get_obs(self):
        """Return the observation (initial weights) for this one-step MDP."""
        return self._obs.copy()

    def sample_action(self, obs=None, deterministic=False):
        """
        Sample a DMP weight vector.

        During warmup (episodes < learning_starts): uniform random
        within action bounds.
        After warmup: SB3 SAC policy with entropy-driven exploration.

        Parameters
        ----------
        obs : np.ndarray or None
            Observation. None uses the initial weights.
        deterministic : bool
            If True, return mean action (no exploration noise).

        Returns
        -------
        weights : np.ndarray, shape (num_weights,)
        """
        if obs is None:
            obs = self._obs

        if self.episodes_done < self.learning_starts and not deterministic:
            return self.dummy_env.action_space.sample()

        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action.flatten()

    def store_transition(self, obs, action, reward, next_obs, done):
        """
        Store one episode result in SB3's replay buffer.

        For this one-step MDP:
            obs      = initial_weights (the starting trajectory)
            action   = DMP weight vector
            reward   = total rollout reward
            next_obs = initial_weights
            done     = True (always, single-step episode)

        SB3 ReplayBuffer expects specific array shapes, so we
        reshape accordingly.
        """
        self.model.replay_buffer.add(
            obs=np.array([obs]),
            next_obs=np.array([next_obs]),
            action=np.array([action]),
            reward=np.array([reward]),
            done=np.array([done]),
            infos=[{}],
        )
        self.episodes_done += 1
        # Keep SB3's internal counter in sync
        self.model.num_timesteps = self.episodes_done

    def update(self):
        """
        Run SB3's SAC gradient steps if enough data is collected.

        Returns
        -------
        updated : bool
            True if gradient steps were taken.
        """
        if self.episodes_done < self.learning_starts:
            return False
        if self.model.replay_buffer.size() < self.batch_size:
            return False

        self.model.train(
            batch_size=self.batch_size,
            gradient_steps=self.gradient_steps,
        )
        return True

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
        Save full SB3 model + replay buffer + best tracking.

        Creates:
            {path}.zip          — SB3 model (policy + critic + optimizer)
            {path}_buffer.pkl   — replay buffer
            {path}_meta.npz     — best weights, reward, episode count
        """
        self.model.save(path)
        self.model.save_replay_buffer(path + "_buffer")
        np.savez(
            path + "_meta.npz",
            best_weights=self.best_weights if self.best_weights is not None else np.array([]),
            best_reward=float(self.best_reward),
            episodes_done=self.episodes_done,
        )

    def load(self, path):
        self.model = SAC.load(path, env=self.dummy_env)

        buffer_path = path + "_buffer.pkl"
        if os.path.exists(buffer_path):
            self.model.load_replay_buffer(path + "_buffer")

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
