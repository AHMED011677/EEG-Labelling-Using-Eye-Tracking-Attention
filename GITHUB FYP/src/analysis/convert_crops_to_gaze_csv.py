import json
import csv
from pathlib import Path
import numpy as np

INPUT_FILE  = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json")
OUTPUT_FILE = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gaze_clusters.csv")

WINDOW_SIZE_S = 120

MIN_FIXATIONS_FOR_ANALYSIS = 2   
FOCUSED_MAX_SPREAD         = 60  
SPARSE_MAX_FIXATIONS       = 3   



def get_xy(fixation: dict) -> tuple[float, float] | None:
    """Extract (x, y) from a fixation dict, trying common key names."""
    x_keys = ["x", "fixation_x", "gaze_x", "norm_x"]
    y_keys = ["y", "fixation_y", "gaze_y", "norm_y"]
    for x_key in x_keys:
        for y_key in y_keys:
            if x_key in fixation and y_key in fixation:
                return float(fixation[x_key]), float(fixation[y_key])
    return None


def get_cluster_id_from_crop(crop: dict) -> int:
    fixations = crop.get("fixations", [])
    points = [xy for f in fixations if (xy := get_xy(f)) is not None]
    n = len(points)

    if n < MIN_FIXATIONS_FOR_ANALYSIS:
        return 2  

    pts = np.array(points)
    spread = (pts[:, 0].std() + pts[:, 1].std()) / 2.0

    if spread < 230:
        return 0  
    elif spread > 270:
        return 1  
    else:
        return 2  


def get_timestamp(crop: dict, fallback_index: int) -> int:
    """Use the crop's own timestamp if available, else compute from index."""
    for key in ["timestamp", "timestamp_ms", "start_time", "time_ms"]:
        if key in crop:
            return int(crop[key])
    return fallback_index * WINDOW_SIZE_S * 1000


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    crops = (
        data
        if isinstance(data, list)
        else data.get("crops", data.get("sessions", []))
    )

    rows = []
    for i, crop in enumerate(crops):
        crop_id    = crop.get("crop_id", crop.get("session_id", f"crop_{i}"))
        timestamp  = get_timestamp(crop, i)
        cluster_id = get_cluster_id_from_crop(crop)
        rows.append({"timestamp": timestamp, "cluster_id": cluster_id, "crop_id": crop_id})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "cluster_id", "crop_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {OUTPUT_FILE}")

    counts = {}
    for row in rows:
        counts[row["cluster_id"]] = counts.get(row["cluster_id"], 0) + 1

    labels = {0: "Focused", 1: "Scattered", 2: "Sparse"}
    for cid, count in sorted(counts.items()):
        print(f"  Cluster {cid} ({labels.get(cid, '?')}): {count} crops")


if __name__ == "__main__":
    main()