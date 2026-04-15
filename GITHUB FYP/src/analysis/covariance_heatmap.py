import json
import numpy as np
import matplotlib.pyplot as plt

with open(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json") as f:
    data = json.load(f)

results = []
all_mean_vals = []
all_std_vals = []

for crop in data["crops"]:
    covs = [np.array(c) for c in crop["gmm"]["covariances"]]
    covs = np.stack(covs)

    mean_cov = np.mean(covs, axis=0)
    std_cov = np.std(covs, axis=0)

    results.append((crop["crop_id"], mean_cov, std_cov))
    all_mean_vals.extend(mean_cov.flatten())
    all_std_vals.extend(std_cov.flatten())

mean_vmin, mean_vmax = min(all_mean_vals), max(all_mean_vals)
std_vmin, std_vmax = min(all_std_vals), max(all_std_vals)

for crop_id, mean_cov, std_cov in results:
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axs[0].imshow(mean_cov, cmap="viridis", vmin=mean_vmin, vmax=mean_vmax)
    axs[0].set_title("Mean Covariance")
    axs[0].set_xticks([0, 1])
    axs[0].set_yticks([0, 1])
    axs[0].set_xticklabels(["X", "Y"])
    axs[0].set_yticklabels(["X", "Y"])

    im1 = axs[1].imshow(std_cov, cmap="magma", vmin=std_vmin, vmax=std_vmax)
    axs[1].set_title("Std Covariance")
    axs[1].set_xticks([0, 1])
    axs[1].set_yticks([0, 1])
    axs[1].set_xticklabels(["X", "Y"])
    axs[1].set_yticklabels(["X", "Y"])

    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, f"{mean_cov[i, j]:.0f}", ha="center", va="center", color="white")
            axs[1].text(j, i, f"{std_cov[i, j]:.0f}", ha="center", va="center", color="white")

    plt.colorbar(im0, ax=axs[0])
    plt.colorbar(im1, ax=axs[1])

    plt.suptitle(f"Covariance - {crop_id}")
    plt.tight_layout()
    plt.show()