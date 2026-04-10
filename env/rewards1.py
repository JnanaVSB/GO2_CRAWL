"""
Reward and termination functions for Go2 Crawl.

These operate on observations from the env. The env returns a 37-dim obs:
    [0:12]   joint_pos
    [12:24]  joint_vel
    [24:27]  base_pos (x, y, z)
    [27:31]  base_quat (w, x, y, z)
    [31:34]  base_linvel
    [34:37]  base_angvel

Reward and termination functions are independent of the env.
They only look at observations.
"""

import numpy as np


def forward_reward(prev_obs, obs, config):
    """
    Reward forward (+x) motion, penalize lateral drift and joint velocity.
    """
    prev_x = prev_obs[24]
    prev_y = prev_obs[25]
    x = obs[24]
    y = obs[25]
    joint_vel = obs[12:24]

    dx = x - prev_x
    dy = y - prev_y

    reward = (
        config["reward_forward_weight"] * dx
        + config["reward_lateral_weight"] * abs(dy)
        + config["reward_joint_vel_weight"] * np.mean(np.abs(joint_vel))
    )

    return reward


def crawl_termination(obs, config):
    """
    Terminate if robot falls, flips, or goes unstable.
    """
    z = obs[26]

    if z < config["min_height"] or z > config["max_height"]:
        return True

    # Quaternion from obs[27:31] in (w, x, y, z) convention.
    # This matches MuJoCo's qpos layout for free joints (qpos[3:7] = w,x,y,z).
    # If the simulator or obs packing order changes, this MUST be updated.
    w, x, y, zz = obs[27:31]
    roll = np.arctan2(2.0 * (w * x + y * zz), 1.0 - 2.0 * (x * x + y * y))
    sinp = 2.0 * (w * y - zz * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1 else np.arcsin(sinp)

    max_roll = np.deg2rad(config["max_abs_roll_deg"])
    max_pitch = np.deg2rad(config["max_abs_pitch_deg"])

    if abs(roll) > max_roll or abs(pitch) > max_pitch:
        return True

    if not np.isfinite(obs).all():
        return True

    return False


REWARD_REGISTRY = {
    "forward": forward_reward,
}

TERMINATION_REGISTRY = {
    "crawl": crawl_termination,
}


def get_reward_fn(name):
    if name not in REWARD_REGISTRY:
        available = ", ".join(REWARD_REGISTRY.keys())
        raise KeyError(f"Unknown reward '{name}'. Available: {available}")
    return REWARD_REGISTRY[name]


def get_termination_fn(name):
    if name not in TERMINATION_REGISTRY:
        available = ", ".join(TERMINATION_REGISTRY.keys())
        raise KeyError(f"Unknown termination '{name}'. Available: {available}")
    return TERMINATION_REGISTRY[name]