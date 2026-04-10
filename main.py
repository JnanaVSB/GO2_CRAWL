import yaml
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add parent directory to path so that dmp/ (which lives outside refactor/) is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmp.dmp_rhythmic import DMPs_rhythmic
from env.go2_env import Go2CrawlEnv
from policy.dmp_policy import DMPPolicy
from agent.cmaes_agent import CMAESAgent
from runner import evolutionary_runner
from runner import sac_dmp_runner
from runner import ppo_dmp_runner
from runner import props_dmp_runner
import shutil


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

    if args.fresh:
        logdir = config["training"]["logdir"]
        # Clear checkpoints
        ckpt_dir = os.path.join(logdir, "checkpoints")
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        # Clear plots
        plots_dir = os.path.join(logdir, "plots")
        if os.path.exists(plots_dir):
            shutil.rmtree(plots_dir)
        # Clear reasoning logs (ProPS)
        reasoning_dir = os.path.join(logdir, "reasoning")
        if os.path.exists(reasoning_dir):
            shutil.rmtree(reasoning_dir)
        # Clear best weights
        best_w = os.path.join(logdir, "best_weights.npy")
        if os.path.exists(best_w):
            os.remove(best_w)
        print(f"Cleared checkpoints, plots, and reasoning in: {logdir}")
        print(f"Log file preserved.")

    ensure_weights(config)

    runner_type = config["runner"]

    if runner_type == "evolutionary":
        run_evolutionary(config)
    elif runner_type == "rl":
        run_rl(config)
    elif runner_type == "props":
        run_props(config)
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")


def ensure_weights(config):
    policy_cfg = config["policy"]
    agent_cfg = config["agent"]

    initial_weights_path = agent_cfg["initial_weights_path"]
    dmp_params_path = policy_cfg["dmp_params_path"]

    if os.path.exists(initial_weights_path) and os.path.exists(dmp_params_path):
        print(f"Found existing weights: {initial_weights_path}")
        print(f"Found existing DMP params: {dmp_params_path}")
        return

    print("Initial weights not found. Generating from dataset...")

    csv_path = policy_cfg["csv_path"]
    n_components = policy_cfg["n_components"]
    n_bfs = policy_cfg["n_bfs"]

    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype(np.float64)
    print(f"Dataset: {X.shape[0]} poses, {X.shape[1]} joints")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_mean = np.mean(X_pca, axis=0)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")

    radius = 0.4
    n_points = 240
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    circle_pca = np.column_stack([
        pca_mean[0] + radius * np.cos(theta),
        pca_mean[1] + radius * np.sin(theta),
    ])

    dmp = DMPs_rhythmic(
        n_dmps=n_components,
        n_bfs=n_bfs,
        ay=np.ones(n_components) * 10.0,
    )
    dmp.imitate_path(y_des=circle_pca.T)
    print(f"DMP weights shape: {dmp.w.shape}")

    os.makedirs(os.path.dirname(initial_weights_path), exist_ok=True)
    np.save(initial_weights_path, dmp.w)
    print(f"Saved: {initial_weights_path}")

    os.makedirs(os.path.dirname(dmp_params_path), exist_ok=True)
    np.savez(dmp_params_path, weights=dmp.w, c=dmp.c, h=dmp.h, goal=dmp.goal)
    print(f"Saved: {dmp_params_path}")


def run_evolutionary(config):
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
    )

    evolutionary_runner.run_training_loop(
        env_cls=Go2CrawlEnv,
        env_kwargs=env_cfg,
        policy_cls=DMPPolicy,
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


def run_rl(config):
    env_cfg = config["env"]
    policy_cfg = config["policy"]
    agent_cfg = config["agent"]
    reward_cfg = config["reward"]
    train_cfg = config["training"]

    agent_type = agent_cfg["type"]
    num_weights = policy_cfg["n_components"] * policy_cfg["n_bfs"]
    initial_weights = np.load(agent_cfg["initial_weights_path"]).flatten()

    if agent_type == "sac_dmp":
        from agent.sac_dmp_agent import SACDMPAgent

        agent = SACDMPAgent(
            num_weights=num_weights,
            initial_weights=initial_weights,
            total_episodes=train_cfg["total_episodes"],
            action_range=agent_cfg.get("action_range", 2.0),
            learning_rate=agent_cfg.get("learning_rate", 3e-4),
            batch_size=agent_cfg.get("batch_size", 256),
            buffer_size=agent_cfg.get("buffer_size", 10_000),
            learning_starts=agent_cfg.get("learning_starts", 100),
            tau=agent_cfg.get("tau", 0.005),
            gamma=agent_cfg.get("gamma", 0.99),
            ent_coef=agent_cfg.get("ent_coef", "auto"),
            target_entropy=agent_cfg.get("target_entropy", "auto"),
            gradient_steps=agent_cfg.get("gradient_steps", 1),
            net_arch=agent_cfg.get("net_arch", [256, 256]),
            seed=agent_cfg.get("seed", None),
            device=agent_cfg.get("device", "auto"),
        )

    elif agent_type == "ppo_dmp":
        from agent.ppo_dmp_agent import PPODMPAgent

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

    else:
        raise ValueError(f"Unknown RL agent type: {agent_type}")

    # Dispatch to the appropriate runner
    if agent_type == "sac_dmp":
        sac_dmp_runner.run_training_loop(
            env_cls=Go2CrawlEnv,
            env_kwargs=env_cfg,
            policy_cls=DMPPolicy,
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

    elif agent_type == "ppo_dmp":
        ppo_dmp_runner.run_training_loop(
            env_cls=Go2CrawlEnv,
            env_kwargs=env_cfg,
            policy_cls=DMPPolicy,
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


def run_props(config):
    """Dispatch for ProPS / ProPS+ runner."""
    from agent.props_dmp_agent import ProPSDMPAgent

    env_cfg = config["env"]
    policy_cfg = config["policy"]
    agent_cfg = config["agent"]
    reward_cfg = config["reward"]
    train_cfg = config["training"]

    num_weights = policy_cfg["n_components"] * policy_cfg["n_bfs"]
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
        policy_cls=DMPPolicy,
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