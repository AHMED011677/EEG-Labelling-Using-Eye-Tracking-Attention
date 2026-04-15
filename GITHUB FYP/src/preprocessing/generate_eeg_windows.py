import json
import mne
import numpy as np
from pathlib import Path

EDF_DIR = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\eeg_images")
CROPS_FILE = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json")
OUTPUT_FILE = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\eeg_windows.npz")

WINDOW_SIZE_S = 120


def main():
    with open(CROPS_FILE, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        crops = data.get("crops", [])
    elif isinstance(data, list):
        crops = data
    else:
        print("Unexpected JSON format in cropped_fixations.json")
        return

    if not crops:
        print("No crops found.")
        return

    edf_cache = {}
    windows = []
    timestamps = []
    crop_ids = []
    session_ids = []

    for crop in crops:
        session_id = crop.get("session_id")
        crop_id = crop.get("crop_id")

        if session_id is None or crop_id is None:
            print(f"Skipping malformed crop: {crop}")
            continue

        edf_path = EDF_DIR / f"{session_id}.edf"

        if not edf_path.exists():
            print(f"Missing EDF file: {edf_path}")
            continue

        if session_id not in edf_cache:
            print(f"Loading EDF: {edf_path.name}")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            edf_cache[session_id] = raw
        else:
            raw = edf_cache[session_id]

        sfreq = int(raw.info["sfreq"])

        crop_index = int(crop.get("crop_index", 0))
        start_time_s = float(crop.get("crop_start_time", crop_index * WINDOW_SIZE_S))
        end_time_s = float(crop.get("crop_end_time", start_time_s + WINDOW_SIZE_S))

        start_sample = int(start_time_s * sfreq)
        end_sample = int(end_time_s * sfreq)

        total_samples = raw.n_times
        if end_sample > total_samples:
            print(f"Skipping {crop_id}: exceeds EDF length")
            continue

        window = raw.get_data(start=start_sample, stop=end_sample)

        windows.append(window)
        timestamps.append(int(start_time_s * 1000))
        crop_ids.append(crop_id)
        session_ids.append(session_id)

        print(f"{crop_id}: extracted window {window.shape} | start={start_time_s}s")

    if not windows:
        print("No EEG windows were extracted.")
        return

    windows = np.array(windows, dtype=np.float32)
    timestamps = np.array(timestamps)
    crop_ids = np.array(crop_ids)
    session_ids = np.array(session_ids)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_FILE,
        windows=windows,
        timestamps=timestamps,
        crop_ids=crop_ids,
        session_ids=session_ids
    )

    print("\nEEG window extraction complete")
    print("Windows shape:", windows.shape)
    print("Timestamps shape:", timestamps.shape)
    print("First 10 timestamps:", timestamps[:10])
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()