import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from pathlib import Path

FIXATIONS_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"

OUTPUT_DIR = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(FIXATIONS_FILE, "r") as f:
    data = json.load(f)

MAX_K = 5

for crop in data["crops"]:
    fixations = crop["fixations"]

    X = np.array([[f["x"], f["y"]] for f in fixations])

    if len(X) < 5:
        continue

    bic_scores = []
    k_values = list(range(1, MAX_K + 1))

    for k in k_values:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bic = gmm.bic(X)
        bic_scores.append(bic)

    best_k = k_values[np.argmin(bic_scores)]

    plt.figure(figsize=(5, 4))
    plt.plot(k_values, bic_scores, marker='o')
    plt.axvline(best_k, linestyle='--', label=f'Best K = {best_k}')

    plt.title(f"BIC Curve - {crop['crop_id']}")
    plt.xlabel("K (components)")
    plt.ylabel("BIC")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"bic_analysis_{crop['crop_id']}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.close()