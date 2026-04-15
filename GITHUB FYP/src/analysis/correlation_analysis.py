import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

GMM_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"

OUTPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\correlation_analysis.png"
)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(GMM_FILE, "r") as f:
    data = json.load(f)

crop_ids = []
var_x_list = []
var_y_list = []
corr_list = []

for crop in data["crops"]:
    covs = [np.array(c) for c in crop["gmm"]["covariances"]]
    covs = np.stack(covs)

    mean_cov = np.mean(covs, axis=0)

    var_x = mean_cov[0, 0]
    var_y = mean_cov[1, 1]
    cov_xy = mean_cov[0, 1]

    if var_x > 0 and var_y > 0:
        corr = cov_xy / np.sqrt(var_x * var_y)
    else:
        corr = 0.0

    crop_ids.append(crop["crop_id"])
    var_x_list.append(var_x)
    var_y_list.append(var_y)
    corr_list.append(corr)

var_x_norm = np.array(var_x_list) / max(var_x_list)
var_y_norm = np.array(var_y_list) / max(var_y_list)

plt.figure(figsize=(12, 6))
plt.plot(crop_ids, var_x_norm, marker='o', label='Variance X (normalised)')
plt.plot(crop_ids, var_y_norm, marker='o', label='Variance Y (normalised)')
plt.plot(crop_ids, corr_list, marker='o', label='Correlation')

plt.axhline(0, linestyle='--')
plt.xticks(rotation=45, ha="right")
plt.ylabel("Value")
plt.xlabel("Crop")
plt.title("Gaze Behaviour Over Time (Variance + Correlation)")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.show()