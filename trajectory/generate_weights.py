"""
Generate initial DMP weights from the pose dataset.

Reads the policy section of a config YAML, fits PCA on the dataset,
builds a circular trajectory in PCA space, trains a rhythmic DMP
to imitate it, and saves the weights and DMP params.

Usage:
    python generate_weights.py --config configs/cmaes_bfs10.yaml
"""

import yaml
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys
import os
sys.path.append("/Volumes/DEV/IRL/Go2_crawl/")
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
        "--circle_radius",
        type=float,
        default=0.4,
        help="Radius of circle trajectory in PCA space",
    )
    parser.add_argument(
        "--circle_points",
        type=int,
        default=240,
        help="Number of points on the circle",
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

    # Build circle in PCA space
    theta = np.linspace(0, 2 * np.pi, args.circle_points, endpoint=False)
    circle_pca = np.column_stack([
        pca_mean[0] + args.circle_radius * np.cos(theta),
        pca_mean[1] + args.circle_radius * np.sin(theta),
    ])
    print(f"Circle: radius={args.circle_radius}, points={args.circle_points}")

    # Train DMP to imitate circle
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

    # Save DMP params (c, h, goal)
    os.makedirs(os.path.dirname(dmp_params_path), exist_ok=True)
    np.savez(
        dmp_params_path,
        weights=dmp.w,
        c=dmp.c,
        h=dmp.h,
        goal=dmp.goal,
    )
    print(f"Saved: {dmp_params_path}")


if __name__ == "__main__":
    main()