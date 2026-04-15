import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier

from modelling.classify import extract_features


def evaluate(
    model: RandomForestClassifier,
    le: LabelEncoder,
    X: np.ndarray,
    y: list[str],
    output_dir: str = "results",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[evaluate] Extracting features for evaluation set...")
    X_feat = extract_features(X)
    y_enc = le.transform(y)
    classes = list(le.classes_)

    y_pred = model.predict(X_feat)
    y_prob = model.predict_proba(X_feat)

    acc = accuracy_score(y_enc, y_pred)
    report = classification_report(y_enc, y_pred, target_names=classes)

    print(f"[evaluate] Accuracy: {acc:.4f}")
    print(report)

    report_path = out / "classification_report.txt"
    report_path.write_text(f"Accuracy: {acc:.4f}\n\n{report}")
    print(f"[evaluate] Report saved to {report_path}")

    _plot_confusion_matrix(y_enc, y_pred, classes, out)
    _plot_roc_curves(y_enc, y_prob, classes, out)

    return {"accuracy": acc, "report": report}


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")

    path = out / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[evaluate] Confusion matrix saved to {path}")


def _plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: list[str],
    out: Path,
) -> None:
    n_classes = len(classes)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        _roc_axes(ax, f"ROC - {classes[1]}")

        path = out / "roc_curve.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[evaluate] ROC curve saved to {path}")

    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))

        fig, ax = plt.subplots(figsize=(7, 6))
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.3f})")

        _roc_axes(ax, "ROC Curves (one-vs-rest)")

        path = out / "roc_curves.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[evaluate] ROC curves saved to {path}")


def _roc_axes(ax, title: str) -> None:
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()