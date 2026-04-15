import json
import numpy as np

INPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"


def main():

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    crops = data.get("crops", [])

    all_covariances = []
    component_counts = []
    point_counts = []

    for crop in crops:

        gmm = crop.get("gmm", {})

        num_components = gmm.get("num_components", 0)
        num_points = gmm.get("num_points", 0)
        covariances = gmm.get("covariances", [])

        component_counts.append(num_components)
        point_counts.append(num_points)

        for cov in covariances:
            all_covariances.append(np.array(cov))

    print("Number of crops:", len(crops))
    print("Average number of points per crop:", np.mean(point_counts))
    print("Average number of components per crop:", np.mean(component_counts))

    if all_covariances:

        avg_cov = np.mean(all_covariances, axis=0)

        print("\nAverage covariance matrix:")
        print(avg_cov)

    else:
        print("No covariance matrices found")


if __name__ == "__main__":
    main()