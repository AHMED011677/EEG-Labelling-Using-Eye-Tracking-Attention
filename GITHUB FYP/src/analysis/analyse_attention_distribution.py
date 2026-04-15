import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

GMM_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"
FIGURE_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\covariance_over_time.png"
OUTPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\tables\attention_analysis.txt"

def main():
    with open(GMM_FILE, "r") as f:
        data = json.load(f)

    spreads = []
    crop_ids = []
    lines = []

    header = "ATTENTION DISTRIBUTION ANALYSIS\n"
    print("\n" + header)
    lines.append(header)

    for crop in data["crops"]:
        covs = crop["gmm"]["covariances"]
        crop_spread = []

        for cov in covs:
            cov = np.array(cov)
            eigvals = np.linalg.eigvals(cov)
            spread = np.mean(eigvals)
            crop_spread.append(spread)

        mean_spread = np.mean(crop_spread)
        spreads.append(mean_spread)
        crop_ids.append(crop["crop_id"])

        line = f"{crop['crop_id']} -> spread = {mean_spread:.2f}"
        print(line)
        lines.append(line)

    mean_global = np.mean(spreads)
    std_global = np.std(spreads)

    print("\nGLOBAL ANALYSIS")
    print(f"Mean spread: {mean_global:.2f}")
    print(f"Std spread: {std_global:.2f}")

    lines.append("\nGLOBAL ANALYSIS")
    lines.append(f"Mean spread: {mean_global:.2f}")
    lines.append(f"Std spread: {std_global:.2f}")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(FIGURE_FILE).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(spreads)), spreads, marker="o")
    plt.xticks(range(len(crop_ids)), crop_ids, rotation=45)
    plt.xlabel("Crop Index")
    plt.ylabel("Mean Covariance Spread")
    plt.title("Attention Spread Over Time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nSaved table to {OUTPUT_FILE}")
    print(f"Saved figure to {FIGURE_FILE}")

if __name__ == "__main__":
    main()