import json
import numpy as np
from sklearn.cluster import DBSCAN

INPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
OUTPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cluster_points.json"

EPS = 50
MIN_SAMPLES = 2


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


def cluster_fixations(fixations):

    points = []

    for fixation in fixations:
        xy = get_xy(fixation)
        if xy is not None:
            points.append(xy)

    if len(points) == 0:
        return []

    points = np.array(points)

    if len(points) == 1:
        return [{
            "cluster_id": 0,
            "center_x": float(points[0][0]),
            "center_y": float(points[0][1]),
            "num_points": 1
        }]

    db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
    labels = db.fit_predict(points)

    cluster_points = []
    unique_labels = set(labels)

    for label in unique_labels:

        if label == -1:
            continue

        cluster_members = points[labels == label]
        center = cluster_members.mean(axis=0)

        cluster_points.append({
            "cluster_id": int(label),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "num_points": int(len(cluster_members))
        })

    return cluster_points


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    output = {"crops": []}

    if isinstance(data, dict):
        sessions = data.get("crops", [])
    elif isinstance(data, list):
        sessions = data
    else:
        print("Unexpected JSON format")
        return

    for session in sessions:
        session_id = session.get("session_id")
        session_index = session.get("session_index")

        fixations = session.get("fixations", [])
        clusters = cluster_fixations(fixations)

        output["crops"].append({
            "session_id": session_id,
            "session_index": session_index,
            "num_fixations": len(fixations),
            "num_clusters": len(clusters),
            "cluster_points": clusters
        })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print("Cluster extraction complete")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()