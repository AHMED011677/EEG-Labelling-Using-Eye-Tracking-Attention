import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from modelling.classify import load_model, extract_features

DATASET_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\labelled_dataset.npz"
MODEL_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\models\rf_model.joblib"
OUTPUT_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\metrics\final_metrics.txt"

def main():
    print("[metrics] Loading dataset and model...")
    data = np.load(DATASET_PATH, allow_pickle=True)
    X = data["X"]
    y = data["y"].tolist()

    model, le = load_model(MODEL_PATH)

    print("[metrics] Extracting features...")
    X_feat = extract_features(X)
    y_enc = le.transform(y)

    y_pred = model.predict(X_feat)
    acc = accuracy_score(y_enc, y_pred)
    report = classification_report(y_enc, y_pred, target_names=le.classes_)

    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        f.write(f"Final Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"[metrics] Saved final metrics to {out}")

if __name__ == "__main__":
    main()