import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path

FIXATIONS_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
GMM_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"

TARGET_CROP_ID = "0000103_crop_3"
MIN_DURATION = 0.2  

def draw_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs
    )
    ax.add_patch(ell)

def main():
    with open(FIXATIONS_FILE, "r") as f:
        fix_data = json.load(f)

    with open(GMM_FILE, "r") as f:
        gmm_data = json.load(f)

    crop_fix = next(c for c in fix_data["crops"] if c["crop_id"] == TARGET_CROP_ID)
    crop_gmm = next(c for c in gmm_data["crops"] if c["crop_id"] == TARGET_CROP_ID)

    fixations = crop_fix["fixations"]

    x_filtered = [f["x"] for f in fixations if f["duration"] >= MIN_DURATION]
    y_filtered = [f["y"] for f in fixations if f["duration"] >= MIN_DURATION]

    means = [np.array(m) for m in crop_gmm["gmm"]["means"]]
    covs = [np.array(c) for c in crop_gmm["gmm"]["covariances"]]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(x_filtered, y_filtered, color="blue", alpha=0.75, label="Filtered Fixations")

    for i, (mean, cov) in enumerate(zip(means, covs)):
        draw_ellipse(mean, cov, ax, n_std=1.5, edgecolor="red", linewidth=2, label="GMM Region" if i == 0 else None)
        ax.scatter(mean[0], mean[1], color="red", marker="x", s=100,
                   label="GMM Mean" if i == 0 else None)

    ax.set_title(f"Filtered Gaze + GMM Regions - {TARGET_CROP_ID}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    output_path = Path(
    rf"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\gaze_overlay_{crop_fix['crop_id']}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()