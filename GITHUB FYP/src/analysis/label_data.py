import numpy as np
import pandas as pd

def load_eeg_labels(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_eeg_windows(path: str):
    data = np.load(path, allow_pickle=True)
    return data["windows"], data["timestamps"], data["crop_ids"]

def align_eeg_to_labels(
    eeg_windows: np.ndarray,
    eeg_crop_ids: np.ndarray,
    labels_df: pd.DataFrame,
):
    X, y = [], []

    crop_id_to_idx = {str(cid): i for i, cid in enumerate(eeg_crop_ids)}

    for _, row in labels_df.iterrows():
        crop_id = str(row["crop_id"])
        label = row["label"]

        if crop_id in crop_id_to_idx:
            j = crop_id_to_idx[crop_id]
            X.append(eeg_windows[j])
            y.append(label)
        else:
            print(f"[label_data] cluster {row['cluster_id']} crop_id '{crop_id}' not found - skipped.")

    return np.array(X), np.array(y)

def build_labelled_dataset(
    labels_path: str,
    eeg_path: str,
    output_path: str | None = None,
):
    print("[label_data] Loading EEG labels...")
    labels_df = load_eeg_labels(labels_path)

    print("[label_data] Loading EEG windows...")
    eeg_windows, eeg_timestamps, eeg_crop_ids = load_eeg_windows(eeg_path)

    print("[label_data] EEG windows shape:", eeg_windows.shape)
    print("[label_data] Crop IDs:", eeg_crop_ids[:5])

    print("[label_data] Aligning EEG windows to labels...")
    X, y = align_eeg_to_labels(eeg_windows, eeg_crop_ids, labels_df)

    print(f"[label_data] Labelled dataset: {X.shape[0]} samples, classes={sorted(set(y))}")

    if output_path:
        np.savez(output_path, X=X, y=y)
        print(f"[label_data] Saved to {output_path}")

    return X, y

if __name__ == "__main__":
    LABELS_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\eeg_labels.csv"
    EEG_WINDOWS_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\eeg_windows.npz"
    OUTPUT_PATH = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\labelled_dataset.npz"

    X, y = build_labelled_dataset(
        labels_path=LABELS_PATH,
        eeg_path=EEG_WINDOWS_PATH,
        output_path=OUTPUT_PATH,
    )