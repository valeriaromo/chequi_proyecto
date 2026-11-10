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
    ax.set_title(title, fontsize=8)
    _save(fig, path)


def plot_roc(y_true, y_score, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(title, fontsize=8)
    _save(fig, path)


def plot_roc_overlay(curves, path, subtitle=""):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, y_true, y_score in curves:
        RocCurveDisplay.from_predictions(y_true, y_score, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title(f"ROC curves – all classifiers {subtitle}".strip(), fontsize=9)
    _save(fig, path)


def plot_pred_vs_actual(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, s=4, alpha=0.25)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set(xlabel="Actual", ylabel="Predicted")
    ax.set_title(title, fontsize=8)
    _save(fig, path)


def plot_importance(table, title, path, top=15):
    t = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.3 * len(t) + 1.5))
    ax.barh(t["feature"], t["importance_mean"], xerr=t["importance_std"])
    ax.set_xlabel("Permutation importance (last fold)")
    ax.set_title(title, fontsize=8)
    _save(fig, path)


def plot_importance_heatmap(matrix, title, path, top=20):
    """matrix: filas = features, columnas = modelos (sin columna 'mean')."""
    t = matrix.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(1.6 * len(t.columns) + 4, 0.32 * len(t) + 2))
    im = ax.imshow(t.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(t.columns)), labels=t.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(t.index)), labels=t.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="Permutation importance")
    ax.set_title(title, fontsize=9)
    _save(fig, path)