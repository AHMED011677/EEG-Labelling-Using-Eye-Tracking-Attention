import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

INPUT_FILE = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json")
OUTPUT_FILE = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cluster_centers.json")

NUM_SYNTHETIC_POINTS = 200


def build_gmm_from_saved(means, covariances, weights):
    gmm = GaussianMixture(n_components=len(weights), covariance_type="full")

    gmm.weights_ = np.array(weights, dtype=float)
    gmm.means_ = np.array(means, dtype=float)
    gmm.covariances_ = np.array(covariances, dtype=float)

    gmm.precisions_cholesky_ = np.linalg.cholesky(
        np.linalg.inv(gmm.covariances_)
    )

    return gmm


def parse_crop_index(crop_id):
    try:
        return int(str(crop_id).split("_")[-1])
    except Exception:
        return 0


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    crops = data.get("crops", [])
    if not crops:
        print("No crops found in gmm_results.json")
        return

    all_centers = []
    synthetic_points_all = []

    for i, crop in enumerate(crops):
        session_id = crop.get("session_id", "unknown")
        crop_id = crop.get("crop_id", f"crop_{i}")
        crop_index = parse_crop_index(crop_id)

        gmm_data = crop.get("gmm", {})
        means = gmm_data.get("means", [])
        covariances = gmm_data.get("covariances", [])
        weights = gmm_data.get("weights", [])

        if not means or not covariances or not weights:
            continue

        try:
            gmm = build_gmm_from_saved(means, covariances, weights)
        except Exception as e:
            print(f"Skipping crop {crop_id}: could not rebuild GMM ({e})")
            continue

        for j, mean in enumerate(gmm.means_):
            all_centers.append({
                "cluster_id": len(all_centers) + 1,
                "session_id": session_id,
                "crop_id": crop_id,
                "crop_index": crop_index,
                "component_index": j,
                "x": float(mean[0]),
                "y": float(mean[1])
            })

        try:
            synthetic_points, _ = gmm.sample(NUM_SYNTHETIC_POINTS)
            synthetic_points_all.append(synthetic_points)
        except Exception:
            pass

    if not all_centers:
        print("No cluster centers were extracted.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump({"clusters": all_centers}, f, indent=4)

    print("Cluster centers saved to:", OUTPUT_FILE)
    print("Total centers:", len(all_centers))

    plt.figure(figsize=(7, 6))

    if synthetic_points_all:
        synthetic_points_all = np.vstack(synthetic_points_all)
        plt.scatter(
            synthetic_points_all[:, 0],
            synthetic_points_all[:, 1],
            color="green",
            alpha=0.3,
            s=10,
            label="Synthetic Gaze Points"
        )

    cluster_centers_xy = np.array([[c["x"], c["y"]] for c in all_centers], dtype=float)
    plt.scatter(
        cluster_centers_xy[:, 0],
        cluster_centers_xy[:, 1],
        color="red",
        marker="X",
        s=120,
        label="Cluster Centers"
    )

    plt.title("Synthetic Gaze Data Generated from GMMs")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()