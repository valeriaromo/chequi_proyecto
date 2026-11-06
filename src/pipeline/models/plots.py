"""Figures: confusion matrix, ROC curves, predicted vs actual."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Dry", "Rain"], ax=ax, colorbar=False)
    ax.set_title(title)
    _save(fig, path)


def plot_roc(y_true, y_score, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(title)
    _save(fig, path)


def plot_roc_overlay(curves, path):
    """curves: iterable of (name, y_true, y_score)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, y_true, y_score in curves:
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("ROC curves – all classifiers")
    _save(fig, path)


def plot_pred_vs_actual(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set(xlabel="Actual", ylabel="Predicted", title=title)
    _save(fig, path)
