import json 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

EEG_IMAGE_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\src\analysis\eeg_plot.png"
GMM_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"
FIXATIONS_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"

TARGET_CROP_INDEX = 3



def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

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
    image = load_image(EEG_IMAGE_PATH)

    with open(GMM_FILE, "r") as f:
        gmm_data = json.load(f)

    with open(FIXATIONS_FILE, "r") as f:
        fixation_data = json.load(f)

    gmm_crop = gmm_data["crops"][TARGET_CROP_INDEX]
    crop_id = gmm_crop["crop_id"]
    means = np.array(gmm_crop["gmm"]["means"], dtype=float)
    covs = [np.array(c, dtype=float) for c in gmm_crop["gmm"]["covariances"]]

    matching_crop = None
    for crop in fixation_data["crops"]:
        if crop["crop_id"] == crop_id:
            matching_crop = crop
            break

    if matching_crop is None:
        raise ValueError(f"No matching crop found for {crop_id}")

    points = np.array(
        [[fix["x"], fix["y"]] for fix in matching_crop["fixations"]],
        dtype=float
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=20,
        alpha=0.45,
        color="blue",
        label="Fixations"
    )

    for i, (mean, cov) in enumerate(zip(means, covs), start=1):
        ax.scatter(
            mean[0],
            mean[1],
            color="red",
            marker="X",
            s=140,
            label="GMM Mean" if i == 1 else None
        )
        draw_ellipse(mean, cov, ax, n_std=2.0)

    ax.set_title(f"GMM Fixation Regions Overlaid on EEG - {crop_id}")
    ax.legend()
    ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
        
    