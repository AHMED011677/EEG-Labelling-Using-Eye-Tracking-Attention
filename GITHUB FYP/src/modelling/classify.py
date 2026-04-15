import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Convert raw EEG windows (n, channels, samples) into a 2-D feature matrix.

    Current features per channel:
        mean, std, min, max, peak-to-peak, band power

    Extend this function to add your own features without touching the rest of the pipeline.
    """
    n, channels, samples = X.shape
    features = []

    for trial in X:
        row = []
        for ch in trial:
            row += [
                ch.mean(),
                ch.std(),
                ch.min(),
                ch.max(),
                ch.max() - ch.min(),
            ]

            ps = np.abs(np.fft.rfft(ch)) ** 2
            freqs = np.fft.rfftfreq(samples)
            for low, hi in [(0, 0.08), (0.08, 0.13), (0.13, 0.30), (0.30, 0.50)]:
                mask = (freqs >= low) & (freqs < hi)
                row.append(ps[mask].mean() if mask.any() else 0.0)
        features.append(row)

    return np.array(features, dtype = np.float32)

def train_classifier(
    X: np.ndarray, 
    y: list[str], 
    rf_params: dict | None = None,
    cv_folds: int = 5,
) -> tuple[RandomForestClassifier, LabelEncoder, np.ndarray]:
    """
    Extract features, encode labels, fit a random Forest, and report 
    cross-validated accuracy.

    Returns
    -------
    model        : fitted RandomForestClassifier
    le           : fitted LabelEncoder (needed to decode predictions later)
    X_feat       : feature matrix (n, n_features)
    """
    print("[classify] Extracting features...")
    X_feat = extract_features(X)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"[classify] Classes: {list(le.classes_)}")

    params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": 42,
    }
    if rf_params:
        params.update(rf_params)

    params["class_weight"] = "balanced"
    model = RandomForestClassifier(**params)

    print(f"[classify] Running {cv_folds}-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_feat, y_enc, cv=cv, scoring="accuracy")
    print(f"[classify] CV accura cy: {scores.mean():.3f} +- {scores.std():.3f}")
    model.fit(X_feat, y_enc)
    return model, le, X_feat

def save_model(model, le, path: str) -> None:
    from pathlib import Path
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "label_encoder": le}, path_obj)
    print(f"[classify] Model saved to {path_obj}")

def load_model(path: str) -> tuple[RandomForestClassifier, LabelEncoder]:
    obj = joblib.load(path)
    return obj["model"], obj["label_encoder"] 

    
    
        