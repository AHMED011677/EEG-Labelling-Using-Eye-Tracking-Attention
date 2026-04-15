import json
import numpy as np
from sklearn.mixture import GaussianMixture

INPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
OUTPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"

MAX_COMPONENTS = 4
RANDOM_STATE = 42


def get_xy(fixation):
    possible_x = ["x", "fixation_x", "gaze_x", "norm_x"]
    possible_y = ["y", "fixation_y", "gaze_y", "norm_y"]

    x = None
    y = None

    for key in possible_x:
        if key in fixation:
            x = fixation[key]
            break

    for key in possible_y:
        if key in fixation:
            y = fixation[key]
            break

    if x is None or y is None:
        return None

    return [float(x), float(y)]


def fit_best_gmm(points, max_components=4):
    if len(points) < 2:
        return None

    X = np.array(points, dtype=float)

    best_gmm = None
    best_bic = np.inf
    best_k = None

    max_k = min(max_components, len(X))

    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=RANDOM_STATE
            )
            gmm.fit(X)
            bic = gmm.bic(X)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                best_k = k
        except Exception:
            continue

    if best_gmm is None:
        return None

    return {
        "model": best_gmm,
        "bic": float(best_bic),
        "n_components": int(best_k)
    }


def extract_points_from_fixations(fixations):
    points = []
    for fixation in fixations:
        xy = get_xy(fixation)
        if xy is not None:
            points.append(xy)
    return points


def serialize_gmm_result(gmm_result, num_points):
    if gmm_result is None:
        return {
            "num_points": int(num_points),
            "num_components": 0,
            "bic": None,
            "weights": [],
            "means": [],
            "covariances": []
        }

    gmm = gmm_result["model"]

    return {
        "num_points": int(num_points),
        "num_components": int(gmm_result["n_components"]),
        "bic": float(gmm_result["bic"]),
        "weights": [float(w) for w in gmm.weights_.tolist()],
        "means": [[float(v) for v in mean] for mean in gmm.means_.tolist()],
        "covariances": [
            [[float(v) for v in row] for row in cov]
            for cov in gmm.covariances_.tolist()
        ]
    }


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        crops = data.get("crops", [])
    elif isinstance(data, list):
        crops = data
    else:
        print("Unexpected JSON format")
        return

    output = {"crops": []}

    for crop in crops:
        session_id = crop.get("session_id", "unknown")
        crop_id = crop.get("crop_id", None)
        fixations = crop.get("fixations", [])

        points = extract_points_from_fixations(fixations)
        gmm_result = fit_best_gmm(points, MAX_COMPONENTS)

        output["crops"].append({
            "session_id": session_id,
            "crop_id": crop_id,
            "gmm": serialize_gmm_result(gmm_result, len(points))
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print("GMM fitting complete")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()