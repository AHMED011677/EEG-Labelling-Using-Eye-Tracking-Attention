import json
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from pathlib import Path

FIXATIONS_FILE = r"C:\Users\ahmed\OneDrive\FINAL PROJECT\data\processed\cropped_fixations.json"

OUTPUT_FILE = Path(
    r"C:\Users\ahmed\OneDrive\FINAL PROJECT\results\figures\confusion_matrix_all.png"
)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(FIXATIONS_FILE, "r") as f:
    data = json.load(f)

MIN_DURATION = 0.1

y_true = []
y_pred = []

for crop in data["crops"]:
    fixations = crop["fixations"]

    if len(fixations) < 10:
        continue

    X = np.array([[f["x"], f["y"]] for f in fixations])

    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)

    cluster_means = []
    for k in range(2):
        durations = [f["duration"] for i, f in enumerate(fixations) if labels[i] == k]
        cluster_means.append(np.mean(durations) if durations else 0)

    fixation_cluster = np.argmax(cluster_means)

    for i, f in enumerate(fixations):
        if f["duration"] >= MIN_DURATION:
            y_true.append(1)   # fixation
        else:
            y_true.append(0)   # saccade-like / short

        if labels[i] == fixation_cluster:
            y_pred.append(1)
        else:
            y_pred.append(0)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Saccade", "Fixation"]
)
disp.plot()

plt.title("Confusion Matrix (All Crops)")
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
plt.show()