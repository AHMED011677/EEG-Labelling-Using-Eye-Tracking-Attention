import mne
import matplotlib.pyplot as plt

EDF_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\eeg_images\0000004.edf"
OUTPUT_IMAGE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\eeg_images\eeg_plot.png"

def main():
    raw = mne.io.read_raw_edf(EDF_FILE, preload=False)

    fig = raw.plot(
        duration=10,
        n_channels=21,
        scalings="auto",
        show=False
    )

    fig.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches="tight")
    print(f"EEG plot saved to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()