import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

EEG_IMAGE_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\src\analysis\eeg_plot.png"
CLUSTER_FILE_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\src\analysis\cluster_centers.json"


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_clusters(cluster_file_path):
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Could not find cluster file: {cluster_file_path}")

    with open(cluster_file_path, "r") as f:
        data = json.load(f)

    if "clusters" not in data:
        raise ValueError("cluster_centers.json must contain a 'clusters' key")

    clusters = np.array(data["clusters"], dtype=float)

    if clusters.ndim != 2 or clusters.shape[1] != 2:
        raise ValueError("Clusters must be in the form [[x1, y1], [x2, y2], ...]")

    return clusters


def overlay_clusters(image, clusters):
    plt.figure(figsize=(10, 7))
    plt.imshow(image)

    x = clusters[:, 0]
    y = clusters[:, 1]

    plt.scatter(
        x,
        y,
        color="red",
        marker="X",
        s=180,
        label="Cluster Centres"
    )

    for i, (xn, yn) in enumerate(zip(x, y)):
        plt.text(xn + 5, yn + 5, f"C{i+1}", fontsize=10)

    plt.title("Overlay of Gaze Cluster Centres on EEG Image")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    image = load_image(EEG_IMAGE_PATH)
    clusters = load_clusters(CLUSTER_FILE_PATH)

    print("Loaded cluster centres:")
    print(clusters)

    overlay_clusters(image, clusters)


if __name__ == "__main__":
    main()