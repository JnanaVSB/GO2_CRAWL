"""
Animate PCA trajectory plots from training checkpoints into a GIF.

Collects all pca_*.png files from the plots directory, sorts them
by step number, and stitches them into an animated GIF showing how
the DMP trajectory evolved during training.

Usage:
    python animate_pca.py --config configs/cmaes_bfs10.yaml
    python animate_pca.py --config configs/cmaes_bfs10.yaml --fps 4 --output pca_evolution.gif
"""

import yaml
import argparse
import os
import re
import glob
import imageio.v2 as imageio


def main():
    parser = argparse.ArgumentParser(description="Animate PCA trajectory plots into a GIF")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the config YAML file",
    )
    parser.add_argument(
        "--fps", type=int, default=3,
        help="Frames per second (default: 3, good for reading plot titles)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output filename. Default: pca_trajectory.gif in plots dir",
    )
    parser.add_argument(
        "--hold_last", type=int, default=3,
        help="Repeat the final frame this many extra times (default: 3)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    logdir = config["training"]["logdir"]
    plots_dir = os.path.join(logdir, "plots")

    if not os.path.exists(plots_dir):
        print(f"No plots directory found: {plots_dir}")
        return

    # Find all PCA checkpoint plots (not the final or comparison ones)
    # Matches: pca_gen_0100.png, pca_ep_000050.png, pca_iter_000050.png
    pattern = os.path.join(plots_dir, "pca_*.png")
    all_files = glob.glob(pattern)

    # Filter to only numbered checkpoint plots, extract step number for sorting
    numbered = []
    for f in all_files:
        basename = os.path.basename(f)
        # Skip pca_final.png and pca_comparison.png
        if basename in ("pca_final.png", "pca_comparison.png"):
            continue
        # Extract the number from the filename
        match = re.search(r"(\d+)\.png$", basename)
        if match:
            step = int(match.group(1))
            numbered.append((step, f))

    if len(numbered) < 2:
        print(f"Found {len(numbered)} PCA plots — need at least 2 for animation.")
        return

    # Sort by step number
    numbered.sort(key=lambda x: x[0])

    print(f"Found {len(numbered)} PCA plots in: {plots_dir}")
    print(f"  First: step {numbered[0][0]}")
    print(f"  Last:  step {numbered[-1][0]}")

    # Also include pca_final.png at the end if it exists
    final_path = os.path.join(plots_dir, "pca_final.png")
    if os.path.exists(final_path):
        numbered.append((numbered[-1][0] + 1, final_path))
        print(f"  Including pca_final.png")

    # Read frames
    frames = []
    for step, filepath in numbered:
        frame = imageio.imread(filepath)
        frames.append(frame)

    # Hold the last frame
    for _ in range(args.hold_last):
        frames.append(frames[-1])

    # Save GIF
    output_path = args.output or os.path.join(plots_dir, "pca_trajectory.gif")
    imageio.mimsave(output_path, frames, fps=args.fps, loop=0)

    print(f"Saved: {output_path} ({len(frames)} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()