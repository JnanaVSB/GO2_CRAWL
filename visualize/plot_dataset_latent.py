"""
Plot all dataset poses in PCA latent space.

Use this before deciding circle vs ellipse — it shows where the
dataset actually sits in PC1/PC2, where pca_mean is, and where
the env base pose projects to.

Usage:
    python -m visualize.plot_dataset_latent --config configs/cmaes_bfs10.yaml
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: results/<logdir>/dataset_latent.png)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["policy"]["csv_path"]
    n_components = config["policy"]["n_components"]
    base_angles = np.array(config["env"]["initial_angles"], dtype=np.float64)

    df = pd.read_csv(csv_path)
    labels = df.iloc[:, 0].astype(str).values
    X = df.iloc[:, 1:].values.astype(np.float64)
    print(f"Dataset: {X.shape[0]} poses, {X.shape[1]} joints")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    pca_mean = np.mean(X_pca, axis=0)
    base_latent = pca.transform(base_angles.reshape(1, -1))[0]

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"pca_mean       = {pca_mean}")
    print(f"X_pca[0]       = {X_pca[0]}  (label='{labels[0]}')")
    print(f"base_latent    = {base_latent}")
    print(f"||X_pca[0] - pca_mean|| = {np.linalg.norm(X_pca[0] - pca_mean):.4f}")
    print(f"||base_latent - pca_mean|| = {np.linalg.norm(base_latent - pca_mean):.4f}")

    fig, ax = plt.subplots(figsize=(9, 9))

    # All dataset poses
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=60, c="steelblue",
               zorder=4, label="Dataset poses")
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (X_pca[i, 0], X_pca[i, 1]),
                    xytext=(5, 5), textcoords="offset points", fontsize=7)

    # First pose highlighted
    ax.scatter(X_pca[0, 0], X_pca[0, 1], s=160, facecolors="none",
               edgecolors="darkorange", linewidth=2, zorder=5,
               label=f"X_pca[0] ('{labels[0]}')")

    # pca_mean (the pivot / circle center)
    ax.scatter(pca_mean[0], pca_mean[1], marker="X", s=180, c="crimson",
               zorder=6, label="pca_mean (pivot)")

    # Base pose projection
    ax.scatter(base_latent[0], base_latent[1], marker="*", s=250, c="green",
               zorder=6, label="base pose (env.initial_angles)")

    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title(f"Dataset poses in latent space\n{csv_path}")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend(loc="best")
    plt.tight_layout()

    if args.out is None:
        logdir = config.get("training", {}).get("logdir", "results/_inspect")
        os.makedirs(logdir, exist_ok=True)
        out = os.path.join(logdir, "dataset_latent.png")
    else:
        out = args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    plt.savefig(out, dpi=120)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()