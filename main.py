"""
Go2 Crawl — Main Entry Point.

Reads a config YAML, sets up the agent + trajectory generator + env,
and dispatches to the appropriate runner (CMA-ES, SAC, PPO, ProPS/ProPS+).

The trajectory generator (PCA+DMP, direct DMP, etc.) is selected via
the config field: trajectory.type

Usage:
    python main.py --config configs/cmaes_bfs10.yaml
    python main.py --config configs/sac_dmp_bfs10.yaml --fresh
"""

import yaml
import argparse
import os
import numpy as np
import shutil

from utils import set_global_seed
from env.go2_env import Go2CrawlEnv
from trajectory.pca_dmp import PCADMPTrajectory
from trajectory.direct_dmp import DirectDMPTrajectory
from agent.cmaes_agent import CMAESAgent
from runner import evolutionary_runner
from runner import sac_dmp_runner
from runner import ppo_dmp_runner
from runner import props_dmp_runner


# -----------------------------------------------------------------------
# Trajectory generator registry
# -----------------------------------------------------------------------

TRAJECTORY_REGISTRY = {
    "pca_dmp": PCADMPTrajectory,
    "direct_dmp": DirectDMPTrajectory,
}


def get_trajectory_cls(config):
    """
    Look up the trajectory generator class from config.

    Reads config["trajectory"]["type"]. Falls back to "pca_dmp"
    if the trajectory section is missing (backward compatibility
    with old configs that don't have a trajectory.type field).
    """
    traj_cfg = config.get("trajectory", {})
    traj_type = traj_cfg.get("type", "pca_dmp")

    if traj_type not in TRAJECTORY_REGISTRY:
        available = ", ".join(TRAJECTORY_REGISTRY.keys())
        raise ValueError(
            f"Unknown trajectory type: '{traj_type}'. Available: {available}"
        )

    return TRAJECTORY_REGISTRY[traj_type]


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Clear previous results and start training from scratch",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set global seed for reproducibility (auto-generates if not in config)
    seed = set_global_seed(config.get("seed", None))
    config["seed"] = seed  # store back so it gets logged in training headers

    if args.fresh:
        logdir = config["training"]["logdir"]
        for subdir in ["checkpoints", "plots", "reasoning"]:
            path = os.path.join(logdir, subdir)
            if os.path.exists(path):
                shutil.rmtree(path)
        best_w = os.path.join(logdir, "best_weights.npy")
        if os.path.exists(best_w):
            os.remove(best_w)
        print(f"Cleared checkpoints, plots, and reasoning in: {logdir}")
        print(f"Log file preserved.")

    # Resolve trajectory generator
    trajectory_cls = get_trajectory_cls(config)

    # Let the trajectory generator handle its own weight initialization
    trajectory_cls.ensure_weights(config)

    runner_type = config["runner"]

    if runner_type == "evolutionary":
        run_evolutionary(config, trajectory_cls)
    elif runner_type == "rl":
        run_rl(config, trajectory_cls)
    elif runner_type == "props":
        run_props(config, trajectory_cls)
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")


# -----------------------------------------------------------------------
# Runner dispatchers
# -----------------------------------------------------------------------

def run_evolutionary(config, trajectory_cls):
    env_cfg = config["env"]
    policy_cfg = config["policy"]
    agent_cfg = config["agent"]
    reward_cfg = config["reward"]
    train_cfg = config["training"]

    initial_weights = np.load(agent_cfg["initial_weights_path"]).flatten()

    agent = CMAESAgent(
        initial_weights=initial_weights,
        sigma0=agent_cfg["sigma0"],
        popsize=agent_cfg["popsize"],
        max_generations=train_cfg["max_generations"],
        seed=config.get("seed", None),
    )

    evolutionary_runner.run_training_loop(
        env_cls=Go2CrawlEnv,
        env_kwargs=env_cfg,
        policy_cls=trajectory_cls,
        policy_kwargs=policy_cfg,
        agent=agent,
        max_generations=train_cfg["max_generations"],
        sim_steps=train_cfg["sim_steps"],
        reward_fn_name=reward_cfg["reward_fn"],
        termination_fn_name=reward_cfg["termination_fn"],
        reward_cfg=reward_cfg,
        logdir=train_cfg["logdir"],
        n_workers=train_cfg["n_workers"],
        checkpoint_every=train_cfg["checkpoint_every"],
        log_every=train_cfg["log_every"],
        full_config=config,
    )


def run_rl(config, trajectory_cls):
    env_cfg = config["env"]
    policy_cfg = config["policy"]
    agent_cfg = config["agent"]
    reward_cfg = config["reward"]
    train_cfg = config["training"]

    agent_type = agent_cfg["type"]
    initial_weights = np.load(agent_cfg["initial_weights_path"]).flatten()

    # SAC: uses proper SB3 wrapper env, runner handles everything
    if agent_type == "sac_dmp":
        sac_dmp_runner.run_training_loop(
            env_cls=Go2CrawlEnv,
            env_kwargs=env_cfg,
            policy_cls=trajectory_cls,
            policy_kwargs=policy_cfg,
            agent_cfg=agent_cfg,
            total_episodes=train_cfg["total_episodes"],
            sim_steps=train_cfg["sim_steps"],
            reward_fn_name=reward_cfg["reward_fn"],
            termination_fn_name=reward_cfg["termination_fn"],
            reward_cfg=reward_cfg,
            logdir=train_cfg["logdir"],
            n_workers=train_cfg.get("n_workers", 1),
            checkpoint_every=train_cfg["checkpoint_every"],
            log_every=train_cfg["log_every"],
            initial_weights=initial_weights,
            full_config=config,
        )
        return

    # PPO: still uses the manual runner pattern (to be updated later)
    if agent_type == "ppo_dmp":
        from agent.ppo_dmp_agent import PPODMPAgent

        temp_policy = trajectory_cls(**policy_cfg)
        num_weights = temp_policy.num_params
        del temp_policy

        agent = PPODMPAgent(
            num_weights=num_weights,
            initial_weights=initial_weights,
            total_episodes=train_cfg["total_episodes"],
            action_range=agent_cfg.get("action_range", 2.0),
            learning_rate=agent_cfg.get("learning_rate", 3e-4),
            n_steps=agent_cfg.get("n_steps", 64),
            batch_size=agent_cfg.get("batch_size", 64),
            n_epochs=agent_cfg.get("n_epochs", 10),
            gamma=agent_cfg.get("gamma", 0.99),
            gae_lambda=agent_cfg.get("gae_lambda", 0.95),
            clip_range=agent_cfg.get("clip_range", 0.2),
            ent_coef=agent_cfg.get("ent_coef", 0.0),
            vf_coef=agent_cfg.get("vf_coef", 0.5),
            max_grad_norm=agent_cfg.get("max_grad_norm", 0.5),
            net_arch=agent_cfg.get("net_arch", [256, 256]),
            seed=agent_cfg.get("seed", None),
            device=agent_cfg.get("device", "auto"),
        )

        ppo_dmp_runner.run_training_loop(
            env_cls=Go2CrawlEnv,
            env_kwargs=env_cfg,
            policy_cls=trajectory_cls,
            policy_kwargs=policy_cfg,
            agent=agent,
            total_episodes=train_cfg["total_episodes"],
            sim_steps=train_cfg["sim_steps"],
            reward_fn_name=reward_cfg["reward_fn"],
            termination_fn_name=reward_cfg["termination_fn"],
            reward_cfg=reward_cfg,
            logdir=train_cfg["logdir"],
            n_workers=train_cfg.get("n_workers", 1),
            checkpoint_every=train_cfg["checkpoint_every"],
            log_every=train_cfg["log_every"],
            full_config=config,
        )
        return

    raise ValueError(f"Unknown RL agent type: {agent_type}")


def run_props(config, trajectory_cls):
    """Dispatch for ProPS / ProPS+ runner."""
    from agent.props_dmp_agent import ProPSDMPAgent

    env_cfg = config["env"]
    policy_cfg = config["policy"]
    agent_cfg = config["agent"]
    reward_cfg = config["reward"]
    train_cfg = config["training"]

    # Build a temporary instance to get num_params
    temp_policy = trajectory_cls(**policy_cfg)
    num_weights = temp_policy.num_params
    del temp_policy

    initial_weights = np.load(agent_cfg["initial_weights_path"]).flatten()

    agent = ProPSDMPAgent(
        num_weights=num_weights,
        initial_weights=initial_weights,
        total_iterations=train_cfg["total_iterations"],
        action_range=agent_cfg.get("action_range", 2.0),
        llm_model_name=agent_cfg.get("llm_model_name", "gpt-4o"),
        llm_provider=agent_cfg.get("llm_provider", None),
        ollama_base_url=agent_cfg.get("ollama_base_url", None),
        template_dir=agent_cfg.get("template_dir", "templates"),
        template_name=agent_cfg.get("template_name", "props_numeric.j2"),
        env_description_file=agent_cfg.get("env_description_file", None),
        feedback_mode=agent_cfg.get("feedback_mode", "basic"),
        objective=agent_cfg.get("objective", "cost"),
        buffer_size=agent_cfg.get("buffer_size", 200),
        warmup_episodes=agent_cfg.get("warmup_episodes", 10),
        optimum=agent_cfg.get("optimum", None),
        step_size=agent_cfg.get("step_size", 0.1),
        height_weight=agent_cfg.get("height_weight", 1.0),
        target_height=reward_cfg.get("target_height", 0.17),
        max_retries=agent_cfg.get("max_retries", 5),
        retry_delay=agent_cfg.get("retry_delay", 60),
    )

    props_dmp_runner.run_training_loop(
        env_cls=Go2CrawlEnv,
        env_kwargs=env_cfg,
        policy_cls=trajectory_cls,
        policy_kwargs=policy_cfg,
        agent=agent,
        total_iterations=train_cfg["total_iterations"],
        sim_steps=train_cfg["sim_steps"],
        reward_fn_name=reward_cfg["reward_fn"],
        termination_fn_name=reward_cfg["termination_fn"],
        reward_cfg=reward_cfg,
        logdir=train_cfg["logdir"],
        n_workers=train_cfg.get("n_workers", 1),
        checkpoint_every=train_cfg["checkpoint_every"],
        log_every=train_cfg["log_every"],
        max_parse_retries=agent_cfg.get("max_parse_retries", 3),
        full_config=config,
    )


if __name__ == "__main__":
    main()