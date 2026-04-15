import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

EEG_IMAGE_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\src\analysis\eeg_plot.png"
FIXATIONS_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
FIGURE_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\heatmap_attention.png"

TARGET_CROP_INDEX = 0

def load_fixations():
    with open(FIXATIONS_FILE, "r") as f:
        data = json.load(f)
    return data

def main():
    eeg_img = cv2.imread(EEG_IMAGE_PATH)
    eeg_img = cv2.cvtColor(eeg_img, cv2.COLOR_BGR2RGB)

    height, width, _ = eeg_img.shape

    data = load_fixations()
    crops = data["crops"]
    crop = crops[TARGET_CROP_INDEX]
    fixations = crop["fixations"]

    heatmap = np.zeros((height, width))

    for f in fixations:
        x = int(f["x"])
        y = int(f["y"])

        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1

    heatmap = gaussian_filter(heatmap, sigma=50)

    heatmap = heatmap / heatmap.max()

    plt.figure(figsize=(10, 7))
    plt.imshow(eeg_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title(f"Gaze Heatmap on EEG - Crop {TARGET_CROP_INDEX}")
    plt.axis("off")
    plt.colorbar(label="Attention Intensity")

    Path(FIGURE_FILE).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

    