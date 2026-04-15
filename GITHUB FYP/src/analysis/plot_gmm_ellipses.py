import json
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

GMM_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"
FIXATIONS_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
FIGURE_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\gmm_attention_multiple_crops.png"

TARGET_CROP_INDICES = [0, 1, 2, 3]

def draw_ellipse(mean, cov, ax, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(vals, 1e-9))

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor="red",
        facecolor="none",
        linewidth=2
    )
    ax.add_patch(ellipse)

def main():
    with open(GMM_FILE, "r") as f:
        gmm_data = json.load(f)

    with open(FIXATIONS_FILE, "r") as f:
        fixation_data = json.load(f)

    gmm_crops = gmm_data["crops"]
    fixation_crops = fixation_data["crops"]

    fixation_lookup = {crop["crop_id"]: crop for crop in fixation_crops if "crop_id" in crop}

    valid_crops = []
    for idx in TARGET_CROP_INDICES:
        if 0 <= idx < len(gmm_crops):
            valid_crops.append(gmm_crops[idx])
        else:
            print(f"Skipping crop index {idx}: out of range")

    if not valid_crops:
        print("No valid crops selected.")
        return

    n_crops = len(valid_crops)
    ncols = 2
    nrows = math.ceil(n_crops / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 5 * nrows))
    
    axes = np.array(axes).reshape(-1)

    for ax, gmm_crop in zip(axes, valid_crops):
        crop_id = gmm_crop["crop_id"]
        gmm = gmm_crop["gmm"]

        means = gmm["means"]
        covs = gmm["covariances"]

        matching_crop = fixation_lookup.get(crop_id)

        if matching_crop is None:
            ax.set_title(f"{crop_id}\nNo matching fixation crop found")
            ax.axis("off")
            continue

        points = []
        for fix in matching_crop.get("fixations", []):
            x = fix.get("x")
            y = fix.get("y")
            if x is not None and y is not None:
                points.append([float(x), float(y)])

        points = np.array(points, dtype=float)

        if len(points) > 0:
            ax.scatter(points[:, 0], points[:, 1], s=20, alpha=0.5, label="Fixations")

        for i, (mean, cov) in enumerate(zip(means, covs), start=1):
            mean = np.array(mean)
            cov = np.array(cov)

            ax.scatter(mean[0], mean[1], color="blue", s=100, marker="x", label=f"Mean {i}")
            draw_ellipse(mean, cov, ax, n_std=2.0)

        ax.set_title(f"GMM Attention - {crop_id}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    for ax in axes[n_crops:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()