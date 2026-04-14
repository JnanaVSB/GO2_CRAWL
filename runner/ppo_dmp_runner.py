"""
PPO-DMP Runner.

Uses SB3's model.learn() with a proper Gymnasium wrapper env.
SB3 owns the training loop — action sampling, rollout buffer,
GAE computation, clipping updates are all handled internally.

Same architecture as SAC runner:
    1. Create the PPOSearchEnv wrapper
    2. Validate with check_env
    3. Build the PPO model
    4. Set up callbacks for logging, checkpointing, best weights
    5. Call model.learn()
    6. Save final results
"""

import os
import time
import datetime
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

from agent.ppo_dmp_agent import PPOSearchEnv, build_ppo_model
from evaluation.logging import write_log_header, save_pca_plot, setup_logdir


LOG_COLUMNS = (
    "timestamp,episode,reward,scaled,avg_reward_50,overall_best,"
    "dist_x,avg_dist_x_50,time_sec"
)


# ---------------------------------------------------------------------------
# Callback for logging, checkpointing, and best weights tracking
# ---------------------------------------------------------------------------

class PPOTrainingCallback(BaseCallback):
    """
    Custom callback for PPO training — handles logging, checkpointing,
    best weights tracking, and PCA plot generation.
    """

    def __init__(
        self,
        logdir,
        checkpoint_every,
        log_every,
        policy_cls,
        policy_kwargs,
        full_config,
        verbose=0,
    ):
        super().__init__(verbose)
        self.logdir = logdir
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.policy_cls = policy_cls
        self.policy_kwargs = policy_kwargs
        self.full_config = full_config

        self.checkpoint_dir, self.plots_dir = setup_logdir(logdir)

        # Log file
        self.log_path = os.path.join(logdir, "training_log.csv")
        write_log_header(self.log_path, full_config or {}, LOG_COLUMNS)
        self.log_file = None

        # Tracking
        self.reward_history = []
        self.dist_history = []
        self.ep_start_time = None
        self.ep_count = 0

    def _get_env(self):
        """Access the underlying PPOSearchEnv through SB3's wrapper."""
        return self.training_env.envs[0].unwrapped

    def _on_training_start(self):
        self.log_file = open(self.log_path, "a")
        self.ep_start_time = time.time()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0]
        self.ep_count += 1
        ep = self.ep_count
        reward = info.get("total_reward", 0.0)
        distance_x = info.get("distance_x", 0.0)
        best_reward = info.get("best_reward", -np.inf)
        best_weights = info.get("best_weights", None)

        # Get scaled reward that SB3 sees
        sac_rewards = self.locals.get("rewards", [])
        scaled_reward = float(sac_rewards[0]) if len(sac_rewards) > 0 else reward

        self.reward_history.append(reward)
        self.dist_history.append(distance_x)

        ep_time = time.time() - self.ep_start_time
        self.ep_start_time = time.time()
        now = datetime.datetime.now().isoformat()

        # --- Logging ---
        if ep % self.log_every == 0:
            avg_reward_50 = np.mean(self.reward_history[-50:])
            avg_dist_50 = np.mean(self.dist_history[-50:])

            print(
                f"Ep {ep}: "
                f"reward={reward:.4f}, "
                f"scaled={scaled_reward:.4f}, "
                f"avg_50={avg_reward_50:.4f}, "
                f"best={best_reward:.4f}, "
                f"dist_x={distance_x:.4f}, "
                f"avg_dist_50={avg_dist_50:.4f}, "
                f"time={ep_time:.1f}s"
            )
            self.log_file.write(
                f"{now},{ep},{reward:.6f},{scaled_reward:.6f},"
                f"{avg_reward_50:.6f},{best_reward:.6f},"
                f"{distance_x:.6f},{avg_dist_50:.6f},{ep_time:.2f}\n"
            )
            self.log_file.flush()

        # --- Checkpointing ---
        if ep > 0 and ep % self.checkpoint_every == 0:
            self.model.save(
                os.path.join(self.checkpoint_dir, f"checkpoint_ep_{ep:06d}")
            )
            self.model.save(
                os.path.join(self.checkpoint_dir, "checkpoint_latest")
            )

            # Save full env search state + histories
            env = self._get_env()
            search_state = env.get_search_state()
            np.savez(
                os.path.join(self.checkpoint_dir, "env_state.npz"),
                **search_state,
            )
            np.savez(
                os.path.join(self.checkpoint_dir, "histories.npz"),
                reward_history=np.array(self.reward_history),
                dist_history=np.array(self.dist_history),
            )

            if best_weights is not None:
                np.save(
                    os.path.join(self.logdir, "best_weights.npy"),
                    best_weights,
                )

                save_pca_plot(
                    self.policy_cls, self.policy_kwargs,
                    best_weights, f"Ep {ep}",
                    best_reward, distance_x,
                    os.path.join(self.plots_dir, f"pca_ep_{ep:06d}.png"),
                )

        return True

    def _on_training_end(self):
        if self.log_file:
            self.log_file.close()

        env = self._get_env()
        best_weights = env.best_weights
        best_reward = env.best_reward

        search_state = env.get_search_state()
        np.savez(
            os.path.join(self.checkpoint_dir, "env_state.npz"),
            **search_state,
        )
        np.savez(
            os.path.join(self.checkpoint_dir, "histories.npz"),
            reward_history=np.array(self.reward_history),
            dist_history=np.array(self.dist_history),
        )

        if best_weights is not None:
            np.save(
                os.path.join(self.logdir, "best_weights.npy"),
                best_weights,
            )
            save_pca_plot(
                self.policy_cls, self.policy_kwargs,
                best_weights, "Final",
                best_reward,
                self.dist_history[-1] if self.dist_history else 0.0,
                os.path.join(self.plots_dir, "pca_final.png"),
            )

        self.model.save(
            os.path.join(self.checkpoint_dir, "checkpoint_latest")
        )

        print(f"Training complete. Best reward: {best_reward:.4f}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_training_loop(
    env_cls,
    env_kwargs,
    policy_cls,
    policy_kwargs,
    agent_cfg,
    total_episodes,
    sim_steps,
    reward_fn_name,
    termination_fn_name,
    reward_cfg,
    logdir,
    checkpoint_every,
    log_every,
    initial_weights,
    full_config=None,
    **kwargs,
):
    """
    PPO training loop using SB3's model.learn().

    Parameters
    ----------
    env_cls : class
        Go2CrawlEnv class.
    env_kwargs : dict
        Constructor kwargs for the MuJoCo env.
    policy_cls : class
        Trajectory generator class.
    policy_kwargs : dict
        Constructor kwargs for the trajectory generator.
    agent_cfg : dict
        Agent config section (PPO hyperparameters).
    total_episodes : int
        Total number of episodes.
    sim_steps : int
        Simulation steps per rollout.
    reward_fn_name : str
        Name of reward function.
    termination_fn_name : str
        Name of termination function.
    reward_cfg : dict
        Reward/termination config.
    logdir : str
        Directory for logs, checkpoints, plots.
    checkpoint_every : int
        Save checkpoint every N episodes.
    log_every : int
        Print and log every N episodes.
    initial_weights : np.ndarray
        Initial DMP weights (the safe prior).
    full_config : dict or None
        Full config dict for log header.
    **kwargs
        Absorbs extra args (e.g. n_workers) without error.
    """
    # Create the wrapper env
    env = PPOSearchEnv(
        env_cls=env_cls,
        env_kwargs=env_kwargs,
        policy_cls=policy_cls,
        policy_kwargs=policy_kwargs,
        initial_weights=initial_weights,
        sim_steps=sim_steps,
        reward_fn_name=reward_fn_name,
        termination_fn_name=termination_fn_name,
        reward_cfg=reward_cfg,
        total_episodes=total_episodes,
        action_scale=agent_cfg.get("action_scale", 0.1),
        reward_scale=agent_cfg.get("reward_scale", 10.0),
    )

    # Validate env before training
    try:
        check_env(env, warn=True, skip_render_check=True)
    except Exception as e:
        print(f"Warning: check_env raised: {e}")

    # Reset after check_env may have polluted state
    env._reset_search_state()

    # Build the PPO model
    model = build_ppo_model(env, agent_cfg)

    # Resume from checkpoint if exists
    resumed_episodes = 0
    checkpoint_dir = os.path.join(logdir, "checkpoints")
    latest_ckpt = os.path.join(checkpoint_dir, "checkpoint_latest.zip")
    if os.path.exists(latest_ckpt):
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_latest")
        model = model.load(ckpt_path, env=env)

        env_state_path = os.path.join(checkpoint_dir, "env_state.npz")
        if os.path.exists(env_state_path):
            state = np.load(env_state_path, allow_pickle=True)
            env.load_search_state(state)
            resumed_episodes = env.episodes_done
        else:
            hist_path = os.path.join(checkpoint_dir, "histories.npz")
            if os.path.exists(hist_path):
                hist = np.load(hist_path, allow_pickle=True)
                env.reward_history = hist["reward_history"].tolist()
                env.episodes_done = len(env.reward_history)
                resumed_episodes = env.episodes_done

        print(f"Resumed from episode {resumed_episodes}")

    remaining = total_episodes - resumed_episodes
    if remaining <= 0:
        print("Training already complete.")
        return

    # Set up callback
    callback = PPOTrainingCallback(
        logdir=logdir,
        checkpoint_every=checkpoint_every,
        log_every=log_every,
        policy_cls=policy_cls,
        policy_kwargs=policy_kwargs,
        full_config=full_config,
    )

    print(f"Starting PPO training: {remaining} episodes remaining")

    # Let SB3 handle everything
    model.learn(
        total_timesteps=remaining,
        callback=callback,
        reset_num_timesteps=False,
    )