import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

INPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"


def plot_cov_ellipse(mean, cov, ax, n_std=2.0, **kwargs):
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        **kwargs
    )
    ax.add_patch(ellipse)


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    crop = data["crops"][0]
    gmm_data = crop["gmm"]

    means = np.array(gmm_data["means"])
    covariances = np.array(gmm_data["covariances"])
    weights = np.array(gmm_data["weights"])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(means[:, 0], means[:, 1], marker="X", s=150, color="red", label="Cluster Centres")

    for i, (mean, cov, weight) in enumerate(zip(means, covariances, weights)):
        plot_cov_ellipse(
            mean,
            cov,
            ax,
            n_std=2.0,
            linewidth=2,
            alpha=0.7,
            label=f"Cluster {i+1}" if i == 0 else None
        )
        ax.text(
            mean[0],
            mean[1],
            f" w={weight:.2f}",
            fontsize=9
        )

    ax.set_title("GMM Ellipses and Cluster Centres")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.autoscale()

    plt.show()


if __name__ == "__main__":
    main()