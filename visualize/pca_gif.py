"""
Animate PCA trajectory plots from training checkpoints into a GIF.

Collects all pca_*.png files from the plots directory, sorts them
by step number, and stitches them into an animated GIF showing how
the trajectory evolved during training.

Silently skips if no PCA plots are found (e.g. direct DMP mode).
"""

import os
import re
import glob
import imageio.v2 as imageio


def generate_pca_gif(logdir, fps=3, output_path=None, hold_last=3):
    """
    Stitch PCA checkpoint plots into an animated GIF.

    Parameters
    ----------
    logdir : str
        Training run log directory.
    fps : int
        Frames per second (default: 3).
    output_path : str or None
        Output file path. Default: plots/pca_trajectory.gif in logdir.
    hold_last : int
        Repeat the final frame this many extra times.
    """
    plots_dir = os.path.join(logdir, "plots")

    if not os.path.exists(plots_dir):
        print(f"  No plots directory found: {plots_dir}")
        return

    # Find all PCA checkpoint plots
    # Matches: pca_gen_0100.png, pca_ep_000050.png, pca_iter_000050.png
    pattern = os.path.join(plots_dir, "pca_*.png")
    all_files = glob.glob(pattern)

    # Filter to numbered checkpoint plots, extract step number for sorting
    numbered = []
    for f in all_files:
        basename = os.path.basename(f)
        # Skip pca_final.png and pca_comparison.png
        if basename in ("pca_final.png", "pca_comparison.png"):
            continue
        match = re.search(r"(\d+)\.png$", basename)
        if match:
            step = int(match.group(1))
            numbered.append((step, f))

    if len(numbered) < 2:
        print(f"  Found {len(numbered)} PCA plots — need at least 2 for animation. Skipping.")
        return

    numbered.sort(key=lambda x: x[0])

    print(f"  Found {len(numbered)} PCA plots")
    print(f"    First: step {numbered[0][0]}")
    print(f"    Last:  step {numbered[-1][0]}")

    # Include pca_final.png at the end if it exists
    final_path = os.path.join(plots_dir, "pca_final.png")
    if os.path.exists(final_path):
        numbered.append((numbered[-1][0] + 1, final_path))
        print(f"    Including pca_final.png")

    # Read frames
    frames = []
    for step, filepath in numbered:
        frame = imageio.imread(filepath)
        frames.append(frame)

    # Hold the last frame
    for _ in range(hold_last):
        frames.append(frames[-1])

    # Save GIF
    output_path = output_path or os.path.join(plots_dir, "pca_trajectory.gif")
    imageio.mimsave(output_path, frames, fps=fps, loop=0)

    print(f"  Saved: {output_path} ({len(frames)} frames at {fps} fps)")