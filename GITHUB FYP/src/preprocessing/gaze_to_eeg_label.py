import json
import numpy as np
from pathlib import Path

PLOT_LEFT = 25
PLOT_RIGHT = 1854
PLOT_TOP = 24
PLOT_BOTTOM = 1917

EEG_DURATION_S = 10.0
CROP_WINDOW_S = 120.0

CHANNELS = [
    "FP1", "FP2", "F3", "F4", "C3", "C4",
    "P3", "P4", "O1", "O2", "F7", "F8",
    "T3", "T4", "T5", "T6", "FZ", "PZ",
    "CZ", "A1", "A2"
]
NUM_CHANNELS = len(CHANNELS)

BASE_DIR = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed")
CLUSTER_FILE = BASE_DIR / "cluster_centers.json"
OUTPUT_FILE = BASE_DIR / "eeg_labels.json"


def pixel_to_time(x_pixel):
    x_pixel = np.clip(x_pixel, PLOT_LEFT, PLOT_RIGHT)
    ratio = (x_pixel - PLOT_LEFT) / (PLOT_RIGHT - PLOT_LEFT)
    return round(float(ratio * EEG_DURATION_S), 3)


def pixel_to_channel(y_pixel):
    y_pixel = np.clip(y_pixel, PLOT_TOP, PLOT_BOTTOM)
    ratio = (y_pixel - PLOT_TOP) / (PLOT_BOTTOM - PLOT_TOP)
    channel_index = int(round(ratio * (NUM_CHANNELS - 1)))
    channel_index = int(np.clip(channel_index, 0, NUM_CHANNELS - 1))
    return CHANNELS[channel_index], channel_index


def classify_region(channel_name):
    frontal = {"FP1", "FP2", "F3", "F4", "F7", "F8", "FZ"}
    temporal = {"T3", "T4", "T5", "T6"}
    central = {"C3", "C4", "CZ"}
    parietal = {"P3", "P4", "PZ"}
    occipital = {"O1", "O2"}

    if channel_name in frontal:
        return "frontal"
    elif channel_name in temporal:
        return "temporal"
    elif channel_name in central:
        return "central"
    elif channel_name in parietal:
        return "parietal"
    elif channel_name in occipital:
        return "occipital"
    else:
        return "other"


def main():
    with open(CLUSTER_FILE, "r") as f:
        data = json.load(f)

    clusters = data["clusters"]
    print(f"Loaded {len(clusters)} gaze cluster centres\n")

    labels = []

    print(f"{'Cluster':<8} {'Crop':<10} {'TimeInCrop':<12} {'AbsTime':<10} {'Channel':<8} {'Region'}")

    for i, cluster in enumerate(clusters):
        x = float(cluster["x"])
        y = float(cluster["y"])
        crop_index = int(cluster["crop_index"])
        session_id = cluster.get("session_id")
        crop_id = cluster.get("crop_id")

        time_in_crop_s = pixel_to_time(x)
        absolute_time_s = round(crop_index * CROP_WINDOW_S + time_in_crop_s, 3)

        channel, idx = pixel_to_channel(y)
        region = classify_region(channel)

        label = {
            "cluster_id": i + 1,
            "session_id": session_id,
            "crop_id": crop_id,
            "crop_index": crop_index,
            "pixel_x": round(x, 2),
            "pixel_y": round(y, 2),
            "time_in_crop_s": time_in_crop_s,
            "absolute_time_s": absolute_time_s,
            "channel": channel,
            "channel_index": idx,
            "region": region,
        }
        labels.append(label)

        print(f"C{i+1:<7} {crop_index:<10} {time_in_crop_s:<12.3f} {absolute_time_s:<10.3f} {channel:<8} {region}")

    output = {"num_clusters": len(labels), "labels": labels}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nLabels saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()