import json
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\eeg_images")

OUTPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\output_fixations_all_sessions.json"
)


MIN_FIXATION_POINTS = 6
DISPERSION_THRESHOLD = 80

def find_session_json_files():
    json_files = sorted(DATA_DIR.glob("*.json"))
    valid_files = []

    print(f"Found {len(json_files)} JSON files\n")

    for json_file in json_files:
        edf_file = DATA_DIR / f"{json_file.stem}.edf"

        if edf_file.exists():
            print(f"Matched: {json_file.name}")
            valid_files.append(json_file)
        else:
            print(f"Missing EDF for: {json_file.name}")

    return valid_files


def dispersion(points):
    """
    Dispersion = (max_x - min_x) + (max_y - min_y)
    Simple measure used in the I-DT algorithm.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def extract_fixations_idt(gaze_points):
    """
    Identification by Dispersion Threshold (I-DT) algorithm.

    gaze_points: list of dicts with keys 'x', 'y', 'timestamp'

    Returns a list of fixations, each with:
        - start_time
        - end_time
        - duration
        - x, y (centroid of fixation)
        - num_points
    """
    if len(gaze_points) < MIN_FIXATION_POINTS:
        return []

    fixations = []
    i = 0

    while i <= len(gaze_points) - MIN_FIXATION_POINTS:
        window = gaze_points[i:i + MIN_FIXATION_POINTS]
        coords = [[p["x"], p["y"]] for p in window]

        if dispersion(coords) <= DISPERSION_THRESHOLD:
            j = i + MIN_FIXATION_POINTS

            while j < len(gaze_points):
                coords.append([gaze_points[j]["x"], gaze_points[j]["y"]])

                if dispersion(coords) > DISPERSION_THRESHOLD:
                    coords.pop()
                    break

                j += 1

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]

            start_time = gaze_points[i]["timestamp"]
            end_time = gaze_points[i + len(coords) - 1]["timestamp"]

            fixations.append({
                "start_time": float(start_time),
                "end_time": float(end_time),
                "duration": float(end_time - start_time),
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "num_points": int(len(coords))
            })

            i += len(coords)
        else:
            i += 1

    return fixations


def process_file(filepath):

    with open(filepath, "r") as f:
        data = json.load(f)

    session_id = Path(filepath).stem
    windows = data.get("session", [])

    all_gaze_points = []

    for window in windows:
        gaze_data = window.get("gaze_data", [])

        for point in gaze_data:
            raw = point.get("raw", {})
            x = raw.get("x")
            y = raw.get("y")
            timestamp = point.get("timestamp")

            if x is not None and y is not None and timestamp is not None:
                all_gaze_points.append({
                    "x": float(x),
                    "y": float(y),
                    "timestamp": float(timestamp)
                })

    all_gaze_points.sort(key=lambda p: p["timestamp"])

    fixations = extract_fixations_idt(all_gaze_points)

    print(f"{session_id}: {len(all_gaze_points)} gaze points -> {len(fixations)} fixations")

    return {
        "session_id": session_id,
        "num_gaze_points": len(all_gaze_points),
        "num_fixations": len(fixations),
        "fixations": fixations
    }


def main():
    input_files = find_session_json_files()

    if not input_files:
        print("No valid JSON session files found.")
        return

    print(f"Found {len(input_files)} session files.\n")

    output = {"sessions": []}

    for filepath in input_files:
        print(f"Processing: {filepath}")
        try:
            session = process_file(filepath)
            output["sessions"].append(session)
        except Exception as e:
            print(f"ERROR processing {filepath}: {e}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=4)

    total_fixations = sum(s["num_fixations"] for s in output["sessions"])
    total_points = sum(s["num_gaze_points"] for s in output["sessions"])

    print("\n========== SUMMARY ==========")
    print(f"Sessions processed: {len(output['sessions'])}")
    print(f"Total gaze points: {total_points}")
    print(f"Total fixations: {total_fixations}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
