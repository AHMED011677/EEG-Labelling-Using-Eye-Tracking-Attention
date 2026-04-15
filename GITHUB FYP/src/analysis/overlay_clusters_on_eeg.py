import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

EEG_IMAGE_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\src\analysis\eeg_plot.png"
CLUSTER_FILE_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\src\analysis\cluster_centers.json"

def main():
    image = cv2.imread(EEG_IMAGE_PATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(CLUSTER_FILE_PATH, "r") as f:
        data = json.load(f)

    clusters = np.array(data["clusters"])

    plt.figure(figsize=(8,6))
    plt.imshow(image)

    h, w, _ = image.shape

    x = clusters[:,0]
    y = clusters[:,1]

    x_norm = (x - x.min()) / (x.max() - x.min()) * w
    y_norm = (y - y.min()) / (y.max() - y.min()) * h

    plt.scatter(
        x_norm,
        y_norm,
        color="red",
        marker="X",
        s=150,
        label="Gaze Cluster Centers"
    )

    plt.title("Gaze Attention on EEG Image")
    plt.legend()
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()