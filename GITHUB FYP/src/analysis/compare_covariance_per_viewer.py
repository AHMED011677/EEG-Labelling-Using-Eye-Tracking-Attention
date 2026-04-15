import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

INPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_results.json"
)

FIGURES_DIR = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures"
)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VIEWER_FIG = FIGURES_DIR / "covariance_per_viewer.png"
TIME_FIG = FIGURES_DIR / "gaze_spread_over_time.png"

JSON_OUTPUT = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\covariance_summary.json"
)
JSON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def avg_covariance(covariances):
    if not covariances:
        return None
    return np.mean([np.array(c) for c in covariances], axis=0)


def cov_spread(cov):
    if cov is None:
        return None
    eigenvalues = np.linalg.eigvalsh(np.array(cov))
    return float(np.mean(eigenvalues))


def parse_crop_index(crop_id):
    try:
        return int(crop_id.split("_crop_")[-1])
    except Exception:
        return None


def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    crops = data.get("crops", [])

    viewer_covs = defaultdict(list)
    viewer_spreads = defaultdict(list)
    time_spreads = defaultdict(list)

    for crop in crops:
        session_id = crop.get("session_id", "unknown")
        crop_id = crop.get("crop_id", "")
        gmm = crop.get("gmm", {})
        covariances = gmm.get("covariances", [])
        crop_index = parse_crop_index(crop_id)

        for cov in covariances:
            spread = cov_spread(cov)
            viewer_covs[session_id].append(cov)
            viewer_spreads[session_id].append(spread)
            if crop_index is not None:
                time_spreads[crop_index].append(spread)

    print("=" * 60)
    print("COVARIANCE COMPARISON PER VIEWER")
    print("=" * 60)

    viewer_summary = {}

    for session_id, covs in viewer_covs.items():
        avg_cov = avg_covariance(covs)
        spreads = viewer_spreads[session_id]

        mean_spread = float(np.mean(spreads))
        std_spread = float(np.std(spreads))

        viewer_summary[session_id] = {
            "num_components": len(covs),
            "mean_spread": mean_spread,
            "std_spread": std_spread,
            "avg_covariance": avg_cov.tolist() if avg_cov is not None else None
        }

        print(f"\nViewer: {session_id}")
        print(f"  Number of GMM components : {len(covs)}")
        print(f"  Mean gaze spread         : {mean_spread:.2f}")
        print(f"  Std of gaze spread       : {std_spread:.2f}")
        if avg_cov is not None:
            print("  Average covariance matrix:")
            print(f"    [{avg_cov[0,0]:.1f}  {avg_cov[0,1]:.1f}]")
            print(f"    [{avg_cov[1,0]:.1f}  {avg_cov[1,1]:.1f}]")

    print("\n" + "=" * 60)
    print("COVARIANCE CHANGE OVER TIME (CROP INDEX)")
    print("=" * 60)

    sorted_indices = sorted(time_spreads.keys())
    time_summary = {}

    for idx in sorted_indices:
        spreads = time_spreads[idx]
        mean_s = float(np.mean(spreads))
        time_summary[idx] = mean_s
        print(f"  Crop {idx:3d}  |  mean spread = {mean_s:.2f}  |  n={len(spreads)}")

    all_spreads = [s for spreads in viewer_spreads.values() for s in spreads]

    if all_spreads:
        global_mean = float(np.mean(all_spreads))
        global_std = float(np.std(all_spreads))
        suggested_threshold = global_mean - global_std

        print("\n" + "=" * 60)
        print("SACCADE THRESHOLD SUGGESTION")
        print("=" * 60)
        print(f"  Global mean spread : {global_mean:.2f}")
        print(f"  Global std spread  : {global_std:.2f}")
        print(f"  Suggested threshold: {suggested_threshold:.2f}")
        print(f"  -> Components with spread < {suggested_threshold:.2f} are likely saccades.")
    else:
        suggested_threshold = None

    if viewer_summary:
        fig, ax = plt.subplots(figsize=(8, 4))
        sessions = list(viewer_summary.keys())
        means = [viewer_summary[s]["mean_spread"] for s in sessions]
        stds = [viewer_summary[s]["std_spread"] for s in sessions]

        ax.bar(sessions, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
        ax.set_title("Mean Gaze Spread per Viewer")
        ax.set_xlabel("Session / Viewer")
        ax.set_ylabel("Mean Gaze Spread")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(VIEWER_FIG, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    if sorted_indices:
        fig, ax = plt.subplots(figsize=(10, 4))
        time_vals = [time_summary[i] for i in sorted_indices]

        ax.plot(sorted_indices, time_vals, marker="o", color="darkorange")
        ax.set_title("Gaze Spread Over Time (per 2-min crop)")
        ax.set_xlabel("Crop Index")
        ax.set_ylabel("Mean Gaze Spread")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(TIME_FIG, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    output = {
        "viewer_summary": viewer_summary,
        "time_summary": {str(k): v for k, v in time_summary.items()},
        "suggested_saccade_threshold": suggested_threshold
    }

    with open(JSON_OUTPUT, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved viewer figure to: {VIEWER_FIG}")
    print(f"Saved time figure to:   {TIME_FIG}")
    print(f"Saved JSON summary to:  {JSON_OUTPUT}")


if __name__ == "__main__":
    main()