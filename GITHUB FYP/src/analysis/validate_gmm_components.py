import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.mixture import GaussianMixture

INPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
OUTPUT_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\gmm_validation.json"
FIGURE_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\bic_validation.png"

FIXED_K = 3
MAX_K = 6
RANDOM_STATE = 42

def get_xy(fixation):
    for x_key in ["x", "fixation_x", "gaze_x", "norm_x"]:
        for y_key in ["y", "fixation_y", "gaze_y", "norm_y"]:
            if x_key in fixation and y_key in fixation:
                return [float(fixation[x_key]), float(fixation[y_key])]
    return None

def extract_points(fixations):
    return [xy for f in fixations if (xy := get_xy(f)) is not None]

def fit_gmm_bic(points):
    X = np.array(points)
    bic_scores = {}
    best_k, best_bic = 1, np.inf
    for k in range(1, min(MAX_K, len(X)) + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=RANDOM_STATE)
            gmm.fit(X)
            bic = gmm.bic(X)
            bic_scores[k] = float(bic)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception:
            continue
    return best_k, best_bic, bic_scores

def fit_gmm_fixed(points, k):
    X = np.array(points)
    k = min(k, len(X))
    try:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=RANDOM_STATE)
        gmm.fit(X)
        return {
            "k": k,
            "bic": float(gmm.bic(X)),
            "means": gmm.means_.tolist(),
            "weights": gmm.weights_.tolist(),
            "covariances": gmm.covariances_.tolist()
        }
    except Exception:
        return None

def main():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    crops = data if isinstance(data, list) else data.get("crops", data.get("sessions", []))

    results = []
    all_bic_curves = {}
    match_count = 0

    print("=" * 65)
    print(f"GMM COMPONENT VALIDATION | Fixed K = {FIXED_K}")
    print("=" * 65)
    print(f"{'Crop':<25} {'Auto K':>6} {'Fixed K':>8} {'Match':>6} {'BIC(auto)':>12} {'BIC(fixed)':>12}")
    print("-" * 65)

    for crop in crops:
        crop_id = crop.get("crop_id", crop.get("session_id", "unknown"))
        points = extract_points(crop.get("fixations", []))

        if len(points) < 2:
            continue

        auto_k, auto_bic, bic_scores = fit_gmm_bic(points)
        fixed_result = fit_gmm_fixed(points, FIXED_K)
        fixed_bic = fixed_result["bic"] if fixed_result else None
        match = (auto_k == FIXED_K)

        if match:
            match_count += 1

        all_bic_curves[crop_id] = bic_scores
        fixed_bic_str = f"{fixed_bic:.1f}" if fixed_bic is not None else "N/A"
        print(f"{crop_id:<25} {auto_k:>6} {FIXED_K:>8} {'YES' if match else 'no':>6} {auto_bic:>12.1f} {fixed_bic_str:>12}")

        results.append({
            "crop_id": crop_id,
            "num_points": len(points),
            "auto_k": int(auto_k),
            "fixed_k": int(FIXED_K),
            "match": bool(match),
            "auto_bic": float(auto_bic),
            "fixed_bic": fixed_bic,
            "bic_scores": bic_scores,
            "fixed_gmm": fixed_result
        })

    total = len(results)
    print("-" * 65)

    if total > 0:
        match_pct = 100 * match_count / total
        print(f"\nMatch rate: {match_count}/{total} crops ({match_pct:.1f}%) auto-selected K={FIXED_K}")

        if match_pct >= 70:
            print(f"-> Good agreement. K={FIXED_K} is a reasonable fixed choice.")
        elif match_pct >= 40:
            print(f"-> Moderate agreement. K={FIXED_K} works for some crops but gaze patterns vary.")
        else:
            most_common_k = Counter(r["auto_k"] for r in results).most_common(1)[0][0]
            print(f"-> Low agreement. Most common auto-selected K = {most_common_k}")

    sample_crops = list(all_bic_curves.items())[:6]

    if sample_crops:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for i, (crop_id, bic_scores) in enumerate(sample_crops):
            ks = sorted(bic_scores.keys())
            bics = [bic_scores[k] for k in ks]
            axes[i].plot(ks, bics, marker="o", color="steelblue")
            axes[i].axvline(x=FIXED_K, color="red", linestyle="--", label=f"Fixed K={FIXED_K}")
            auto_k_for_crop = results[i]["auto_k"]
            if auto_k_for_crop in bic_scores:
                axes[i].scatter([auto_k_for_crop], [bic_scores[auto_k_for_crop]],
                                color="green", s=100, zorder=5, label=f"Auto K={auto_k_for_crop}")
            axes[i].set_title(crop_id, fontsize=8)
            axes[i].set_xlabel("K (components)")
            axes[i].set_ylabel("BIC")
            axes[i].legend(fontsize=7)
            axes[i].grid(alpha=0.3)

        for j in range(len(sample_crops), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(f"BIC Curves: Auto K vs Fixed K={FIXED_K}", fontsize=12)
        plt.tight_layout()
        plt.savefig(FIGURE_FILE, dpi=300, bbox_inches="tight")
        plt.show()

    with open(OUTPUT_FILE, "w") as f:
        json.dump({"fixed_k": FIXED_K, "results": results}, f, indent=4)

    print(f"\nSaved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()