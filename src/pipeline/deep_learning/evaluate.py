from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
import params as P


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean())
               for b in range(bins) if (idx == b).any())


def clf_metrics(y, p, clim, mun, dates):
    series, base = clim
    idx = pd.MultiIndex.from_arrays([mun, pd.DatetimeIndex(dates).month])
    p_clim = series.reindex(idx).fillna(base).to_numpy()
    pred = p >= 0.5
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), f1=f1_score(y, pred, zero_division=0),
                roc_auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                bss=1 - brier_score_loss(y, p) / brier_score_loss(y, p_clim), ece=ece(y, p))


def reg_metrics(y, p):
    return dict(mae=mean_absolute_error(y, p), rmse=float(np.sqrt(np.mean((y - p) ** 2))), r2=r2_score(y, p))


def evaluate_fold(fold, pred, targets):
    out = {}
    for i, t in enumerate(targets):
        y, p = fold["Yte"][:, i], pred[:, i]
        if P.TARGETS[t]["kind"] == "clf":
            out[t] = clf_metrics(y, p, fold["clim"][t], fold["mun_te"], fold["dates_te"])
        else:
            mu, sd = fold["yscale"][t]
            out[t] = reg_metrics(y * sd + mu, p * sd + mu)
    return out


def load_baseline(target):
    """Per-year ML baseline metrics from the outputs_fase2 CSV (needs a 'year' column; filter by horizon in params)."""
    spec = P.TARGETS[target]
    path = spec.get("baseline_csv")
    if not path or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    for k, v in spec.get("baseline_filter", {}).items():
        df = df[df[k] == v]
    return df.groupby("year", observed=True).mean(numeric_only=True) if "year" in df.columns else None