import json
from pathlib import Path

WINDOW_SIZE = 120  

INPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\output_fixations_all_sessions.json"
)
OUTPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"
)

def crop_fixations(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    cropped_sessions = []

    for session in data["sessions"]:
        session_id = session["session_id"]
        fixations = session["fixations"]

        if not fixations:
            continue

        min_time = min(float(f["start_time"]) for f in fixations)

        crops = {}

        for fixation in fixations:
            start_time = float(fixation["start_time"])
            relative_time = start_time - min_time 
            crop_index = int(relative_time // WINDOW_SIZE)    
            crop_start_time = crop_index * WINDOW_SIZE
            crop_end_time = crop_start_time + WINDOW_SIZE

            if crop_index not in crops:
                crops[crop_index] = {
                    "session_id": session_id,
                    "crop_id": f"{session_id}_crop_{crop_index}",
                    "crop_index": crop_index,
                    "crop_start_time": crop_start_time,
                    "crop_end_time": crop_end_time,
                    "fixations": []
                }
            crops[crop_index]["fixations"].append(fixation)

        for crop_index in sorted(crops.keys()):
            crop_entry = crops[crop_index]
            crop_entry["num_fixations"] = len(crop_entry["fixations"])
            cropped_sessions.append(crop_entry)

        print(f"{session_id}: {len(crops)} crops")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump({"crops": cropped_sessions}, f, indent=4)

    print("\nCropping complete")
    print(f"Total crops: {len(cropped_sessions)}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    crop_fixations(INPUT_FILE, OUTPUT_FILE)