import json
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\gaze_filtering_plot.png"
)

with open(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json", "r") as f:
    data = json.load(f)

MIN_DURATION = 0.2

for crop in data["crops"]:
    fixations = crop["fixations"]

    x_all = [f["x"] for f in fixations]
    y_all = [f["y"] for f in fixations]

    x_filtered = [f["x"] for f in fixations if f["duration"] >= MIN_DURATION]
    y_filtered = [f["y"] for f in fixations if f["duration"] >= MIN_DURATION]

    x_removed = [f["x"] for f in fixations if f["duration"] < MIN_DURATION]
    y_removed = [f["y"] for f in fixations if f["duration"] < MIN_DURATION]

    plt.figure(figsize=(8, 6))
    plt.scatter(x_all, y_all, color="grey", alpha=0.2, label="All Fixations")
    plt.scatter(x_removed, y_removed, color="red", alpha=0.7, label="Removed (short)")
    plt.scatter(x_filtered, y_filtered, color="blue", alpha=0.8, label="Filtered (kept)")
    plt.savefig(
    rf"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\gaze_filtering_{crop['crop_id']}.png",
    dpi=300,
    bbox_inches='tight'
)

    plt.title(f"Gaze Filtering - {crop['crop_id']}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()