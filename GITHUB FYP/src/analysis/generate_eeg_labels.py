import json
import csv

INPUT_FILE  = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\eeg_labels.json"
OUTPUT_CSV  = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\eeg_labels.csv"

ABNORMAL_REGIONS = {"frontal", "temporal"}

def classify_label(region):
    if region in ABNORMAL_REGIONS:
        return "abnormal"
    else:
        return "normal"

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    labels = data["labels"]

    print(f"Generating labels for {len(labels)} gaze clusters...\n")

    rows = []

    for entry in labels:
        cluster_id = entry["cluster_id"]
        channel = entry["channel"]
        absolute_time_s = entry["absolute_time_s"]
        region = entry["region"]
        pixel_x = entry["pixel_x"]
        pixel_y = entry["pixel_y"]
        channel_index = entry["channel_index"]

        label = classify_label(region)

        row = {
            "cluster_id": cluster_id,
            "session_id": entry["session_id"],
            "crop_id": entry["crop_id"],
            "channel": channel,
            "channel_index": channel_index,
            "absolute_time_s": absolute_time_s,
            "region": region,
            "label": label,
            "pixel_x": pixel_x,
            "pixel_y": pixel_y,
        }
        rows.append(row)

        print(f"Cluster {cluster_id} | {channel} | {absolute_time_s}s | {region} | --> {label}")

    fieldnames = [
        "cluster_id", "session_id", "crop_id", "channel", "channel_index",
        "absolute_time_s", "region", "label", "pixel_x", "pixel_y"
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Labels saved to: {OUTPUT_CSV}")
    print(f"Total clusters labelled : {len(rows)}")
    print(f"Abnormal                : {sum(1 for r in rows if r['label'] == 'abnormal')}")
    print(f"Normal                  : {sum(1 for r in rows if r['label'] == 'normal')}")

if __name__ == "__main__":
    main()