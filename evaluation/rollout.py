"""
Rollout evaluation for Go2 Crawl.

Single source of truth for evaluating a DMP weight vector via MuJoCo rollout.
All runners import evaluate_single() from here instead of duplicating it.

The function creates its own env and policy per call so it is safe for
multiprocessing (each worker gets independent instances).

Returns
-------
EvalResult : namedtuple
    total_reward, steps, terminated, distance_x, avg_height_dev
"""

import numpy as np
from collections import namedtuple

from env.rewards import get_reward_fn, get_termination_fn


EvalResult = namedtuple(
    "EvalResult",
    ["total_reward", "steps", "terminated", "distance_x", "avg_height_dev"],
)


def evaluate_single(args):
    """
    Evaluate one weight vector via MuJoCo rollout.

    Parameters
    ----------
    args : tuple
        (candidate, env_cls, env_kwargs, policy_cls, policy_kwargs,
         sim_steps, reward_fn_name, termination_fn_name, reward_cfg)

        This is a single tuple so the function works with
        multiprocessing.Pool.map.

    Returns
    -------
    EvalResult
        total_reward : float
        steps : int
        terminated : bool
        distance_x : float
        avg_height_dev : float
            Mean absolute deviation of base height from target crawl height.
    """
    (candidate, env_cls, env_kwargs, policy_cls, policy_kwargs,
     sim_steps, reward_fn_name, termination_fn_name, reward_cfg) = args

    env = env_cls(**env_kwargs)
    policy = policy_cls(**policy_kwargs)
    reward_fn = get_reward_fn(reward_fn_name)
    termination_fn = get_termination_fn(termination_fn_name)

    joint_traj = policy.generate_trajectory(candidate)
    traj_len = len(joint_traj)

    prev_obs, _ = env.reset()
    start_x = prev_obs[24]
    total_reward = 0.0
    terminated = False

    # Track height deviation (used by ProPS, harmless overhead for others)
    target_height = reward_cfg.get("target_height", 0.17)
    height_devs = []

    for t in range(sim_steps):
        target = joint_traj[t % traj_len]
        obs, _, _, _, _ = env.step(target)

        reward = reward_fn(prev_obs, obs, reward_cfg)
        total_reward += reward

        # Track height deviation
        z = obs[26]
        height_devs.append(abs(z - target_height))

        terminated = termination_fn(obs, reward_cfg)
        if terminated:
            total_reward -= reward_cfg.get("fall_penalty", 0.0)
            break

        prev_obs = obs

    distance_x = obs[24] - start_x
    avg_height_dev = float(np.mean(height_devs)) if height_devs else 0.0

    return EvalResult(
        total_reward=total_reward,
        steps=t + 1,
        terminated=terminated,
        distance_x=distance_x,
        avg_height_dev=avg_height_dev,
    )


def build_eval_args(candidate, env_cls, env_kwargs, policy_cls, policy_kwargs,
                    sim_steps, reward_fn_name, termination_fn_name, reward_cfg):
    """
    Convenience function to pack evaluation arguments into the tuple
    expected by evaluate_single().
    """
    return (
        candidate, env_cls, env_kwargs, policy_cls, policy_kwargs,
        sim_steps, reward_fn_name, termination_fn_name, reward_cfg,
    )