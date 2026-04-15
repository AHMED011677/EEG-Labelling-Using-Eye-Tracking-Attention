import json
import numpy as np
from pathlib import Path
from collections import defaultdict

GMM_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"
)

OUTPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\tables\covariance_summary.txt"
)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def get_session_id(crop_id: str) -> str:
    return crop_id.split("_crop_")[0]

def format_entry(mean_val: float, std_val: float) -> str:
    return f"{mean_val:.2f} ± {std_val:.2f}"

with open(GMM_FILE, "r") as f:
    data = json.load(f)

session_covariances = defaultdict(list)

for crop in data["crops"]:
    covs = [np.array(c, dtype=float) for c in crop["gmm"]["covariances"]]
    if not covs:
        continue

    mean_cov_for_crop = np.mean(np.stack(covs), axis=0)
    session_id = get_session_id(crop["crop_id"])
    session_covariances[session_id].append(mean_cov_for_crop)

lines = []
lines.append("AVERAGE COVARIANCE MATRIX SUMMARY\n")

for session_id, cov_list in session_covariances.items():
    cov_array = np.stack(cov_list)

    mean_matrix = np.mean(cov_array, axis=0)
    std_matrix = np.std(cov_array, axis=0)

    lines.append(f"Session / Doctor: {session_id}")
    lines.append("Mean covariance matrix with standard deviation:")
    lines.append("[")
    lines.append(
        f"  [{format_entry(mean_matrix[0,0], std_matrix[0,0])}, "
        f"{format_entry(mean_matrix[0,1], std_matrix[0,1])}]"
    )
    lines.append(
        f"  [{format_entry(mean_matrix[1,0], std_matrix[1,0])}, "
        f"{format_entry(mean_matrix[1,1], std_matrix[1,1])}]"
    )
    lines.append("]\n")

report_text = "\n".join(lines)
print(report_text)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(report_text)