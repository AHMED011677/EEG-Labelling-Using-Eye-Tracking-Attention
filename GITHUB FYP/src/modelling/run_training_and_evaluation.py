import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from modelling.classify import train_classifier, save_model
from modelling.evaluate import evaluate

DATASET_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\labelled_dataset.npz"
MODEL_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\models\rf_model.joblib"
OUTPUT_DIR = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results"

def main():
    print("[run] Loading labelled dataset...")
    data = np.load(DATASET_PATH, allow_pickle=True)

    X = data["X"]
    y = data["y"].tolist()

    print(f"[run] X shape: {X.shape}")
    print(f"[run] Number of labels: {len(y)}")

    model, le, _ = train_classifier(X, y, cv_folds=5)
    save_model(model, le, MODEL_PATH)

    results = evaluate(model, le, X, y, output_dir=OUTPUT_DIR)

    print("\n[run] Final results")
    print(f"Accuracy: {results['accuracy']:.4f}")

if __name__ == "__main__":
    main()