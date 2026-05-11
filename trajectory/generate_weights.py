"""
Generate initial DMP weights from the pose dataset.

Reads the policy section of a config YAML, fits PCA on the dataset,
builds an initial trajectory (circle or ellipse) in PCA space, trains
a rhythmic DMP to imitate it, and saves the weights and DMP params.

The initial trajectory is centered at the dataset mean in latent space.
Its start point (DMP at t=0) is the point on the shape nearest to
X_pca[0] (first dataset pose projected into latent space).

The joint-space inverse transform of that start point is also saved
as `start_joints` in the DMP params file, so the env can reset the
robot to the exact configuration the DMP commands at t=0.

Usage:
    python -m trajectory.generate_weights --config configs/cmaes_bfs10.yaml
"""

import yaml
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from dmp.dmp_rhythmic import DMPs_rhythmic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--circle_points",
        type=int,
        default=240,
        help="Number of points on the initial trajectory",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    policy_cfg = config["policy"]
    agent_cfg = config["agent"]

    csv_path = policy_cfg["csv_path"]
    n_components = policy_cfg["n_components"]
    n_bfs = policy_cfg["n_bfs"]
    dmp_params_path = policy_cfg["dmp_params_path"]
    initial_weights_path = agent_cfg["initial_weights_path"]

    # Load dataset
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values.astype(np.float64)
    print(f"Dataset: {X.shape[0]} poses, {X.shape[1]} joints")

    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_mean = np.mean(X_pca, axis=0)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"PCA mean: {pca_mean}")

    # Initial trajectory shape in PCA space, centered at pca_mean.
    # Start point (DMP at t=0) = the point on the shape nearest to X_pca[0].
    init_shape = policy_cfg.get("init_shape", "circle")
    target_latent = X_pca[0]

    if init_shape == "circle":
        # Radius derived from data so X_pca[0] lies exactly on the circle.
        # pca1 / pca2 in the config are ignored.
        radius = float(np.linalg.norm(X_pca[0] - pca_mean))
        v = target_latent - pca_mean
        phi0 = float(np.arctan2(v[1], v[0]))
        theta = np.linspace(0, 2 * np.pi, args.circle_points, endpoint=False) + phi0
        circle_pca = np.column_stack([
            pca_mean[0] + radius * np.cos(theta),
            pca_mean[1] + radius * np.sin(theta),
        ])
        print(f"Init shape: circle, radius={radius:.4f}, "
              f"start_angle={np.degrees(phi0):.2f} deg")

    elif init_shape == "ellipse":
        a = policy_cfg.get("pca1") if policy_cfg.get("pca1") is not None else 0.4
        b = policy_cfg.get("pca2") if policy_cfg.get("pca2") is not None else 0.1
        # Numerical search for the point on the ellipse nearest to X_pca[0].
        theta_dense = np.linspace(0, 2 * np.pi, 4000, endpoint=False)
        pts = np.column_stack([
            pca_mean[0] + a * np.cos(theta_dense),
            pca_mean[1] + b * np.sin(theta_dense),
        ])
        phi0 = float(theta_dense[np.argmin(np.linalg.norm(pts - target_latent, axis=1))])
        theta = np.linspace(0, 2 * np.pi, args.circle_points, endpoint=False) + phi0
        circle_pca = np.column_stack([
            pca_mean[0] + a * np.cos(theta),
            pca_mean[1] + b * np.sin(theta),
        ])
        print(f"Init shape: ellipse, a={a}, b={b}, "
              f"start_angle={np.degrees(phi0):.2f} deg")

    else:
        raise ValueError(f"Unknown init_shape: {init_shape!r}. Use 'circle' or 'ellipse'.")

    # Joint-space start pose: the env should reset to this.
    start_joints = pca.inverse_transform(circle_pca[0].reshape(1, -1))[0]
    print(f"start_joints (env base pose): {start_joints}")

    # Train DMP to imitate the initial shape
    dmp = DMPs_rhythmic(
        n_dmps=n_components,
        n_bfs=n_bfs,
        ay=np.ones(n_components) * 10.0,
    )
    dmp.imitate_path(y_des=circle_pca.T)

    print(f"DMP weights shape: {dmp.w.shape}")
    print(f"DMP goal: {dmp.goal}")

    # Save initial weights
    os.makedirs(os.path.dirname(initial_weights_path), exist_ok=True)
    np.save(initial_weights_path, dmp.w)
    print(f"Saved: {initial_weights_path}")

    # Save DMP params (c, h, goal, start_joints)
    os.makedirs(os.path.dirname(dmp_params_path), exist_ok=True)
    np.savez(
        dmp_params_path,
        weights=dmp.w,
        c=dmp.c,
        h=dmp.h,
        goal=dmp.goal,
        start_joints=start_joints,
    )
    print(f"Saved: {dmp_params_path}")


if __name__ == "__main__":
    main()