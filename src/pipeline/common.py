import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import settings 

OUT = ROOT / "outputs_fase2" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

RAIN_COL = "precipitation_mm" 
RAIN_THRESH = 1.0            # evento = lluvia >= 1 mm
TMIN_COL = "T2M_MIN"
HORIZON = 1
FOLDS = [2023, 2024, 2025, 2026]


def load_master():
    d = Path(settings.MASTER_DIR)
    df = pd.read_parquet(d / "master_table.parquet")
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df:
        df["year"] = df["date"].dt.year
    return df.sort_values(["municipio", "date"]).reset_index(drop=True)


def rain_target(df, h=HORIZON):
    fut = df.groupby("municipio", observed=True)[RAIN_COL].shift(-h)
    return (fut >= RAIN_THRESH).astype(float).where(fut.notna())


def tmin_target(df, h=HORIZON):
    return df.groupby("municipio", observed=True)[TMIN_COL].shift(-h)


def feature_cols(df):
    drop = {"date", "municipio", "year"}
    cols = [c for c in df.columns
            if c not in drop and "aod" not in c.lower() and not c.startswith(("target_", "y_"))]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def train_mask(df, Y, h=HORIZON):
# el target de la última fila (fecha + h) debe caer antes del 1-ene del año Y
    cutoff = pd.Timestamp(year=Y, month=1, day=1)
    return (df["date"] + pd.Timedelta(days=h)) < cutoff

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from dl_validation import cfg
from deep_learning import params as P
from deep_learning import data, evaluate, train


def load():
    df = data.load_master()
    feats = data.feature_cols(df)
    return data.add_targets(data.ffill_features(df, feats), list(P.TARGETS)), feats   # builds every target column once


def run_fold(df, feats, year, targets, seed):
    targets = list(targets)
    fold = data.build_fold(df, feats, year, targets)
    model, hist, secs = train.fit(fold, len(feats), targets, seed)
    return dict(targets=targets, fold=fold, model=model, hist=hist, secs=secs,
                pred=train.predict(model, fold["Xte"], targets), pred_va=train.predict(model, fold["Xva"], targets))


def load_xgb(target):
    b = cfg.BASELINE.get(target)
    if not b or not Path(b["csv"]).exists():
        return None
    name = Path(b["csv"]).name
    df = pd.read_csv(b["csv"])
    df.columns = [c.lower() for c in df.columns]
    metric = b["metric"].lower()
    for k, v in b["filters"].items():
        k = k.lower()
        if k not in df.columns:
            raise ValueError(f"{name}: filter column '{k}' not found; columns are {list(df.columns)}")
        df = df[df[k] == v]
    for k in ("year", metric):
        if k not in df.columns:
            raise ValueError(f"{name}: column '{k}' not found; columns are {list(df.columns)}")
    if df.empty:
        raise ValueError(f"{name}: filters {b['filters']} leave no rows")
    return df.groupby("year", observed=True)[metric].mean()


# ---- same-baseline scoring (fit on years < test year) ----
def smooth_clim(df, target, year):
    spec = P.TARGETS[target]
    fit = df[df.year < year]
    v = fit[spec["col"]]
    if spec["kind"] == "clf":
        v = (v >= spec["thr"]).astype(float).where(v.notna())
    doy = fit.date.dt.dayofyear.clip(upper=365)
    tab = v.groupby([fit.municipio, doy], observed=True).mean().unstack().reindex(columns=range(1, 366))
    arr = tab.to_numpy()
    arr = np.where(np.isnan(arr), np.nanmean(arr, axis=1, keepdims=True), arr)
    k = cfg.CLIM_WINDOW
    pad = np.concatenate([arr[:, -k:], arr, arr[:, :k]], axis=1)          # circular smoothing
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    return pd.DataFrame(np.stack([np.convolve(r, kern, mode="valid") for r in pad]), index=tab.index, columns=range(1, 366))


def clim_lookup(sm, mun, dates):
    doy = np.minimum((pd.DatetimeIndex(dates) + pd.Timedelta(days=P.HORIZON)).dayofyear, 365)
    return sm.to_numpy()[sm.index.get_indexer(mun), doy - 1]


def persistence(df, target, mun, dates):
    s = df.set_index(["municipio", "date"])[P.TARGETS[target]["col"]]
    return s.reindex(pd.MultiIndex.from_arrays([mun, pd.DatetimeIndex(dates)])).to_numpy()


def threshold_metrics(fold, r, i):
    """Decision threshold tuned on the validation year (max F1), applied to the test year."""
    yv, pv = fold["Yva"][:, i], r["pred_va"][:, i]
    grid = np.arange(0.05, 0.96, 0.05)
    thr = float(grid[int(np.argmax([f1_score(yv, pv >= g, zero_division=0) for g in grid]))])
    y, p = fold["Yte"][:, i], r["pred"][:, i]
    return dict(thr=thr, precision=precision_score(y, p >= thr, zero_division=0),
                recall=recall_score(y, p >= thr, zero_division=0), f1=f1_score(y, p >= thr, zero_division=0))


def score(target, r, df, year):
    fold, spec = r["fold"], P.TARGETS[target]
    i = r["targets"].index(target)                                         # column of THIS run's targets
    y, p = fold["Yte"][:, i], r["pred"][:, i]
    pc = clim_lookup(smooth_clim(df, target, year), fold["mun_te"], fold["dates_te"])
    if spec["kind"] == "clf":
        return dict(bss=1 - np.mean((p - y) ** 2) / np.mean((pc - y) ** 2), auc=roc_auc_score(y, p),
                    ece=evaluate.ece(y, p), **threshold_metrics(fold, r, i))
    mu, sd = fold["yscale"][target]
    y, p = y * sd + mu, p * sd + mu
    pp = persistence(df, target, fold["mun_te"], fold["dates_te"])
    ok = ~np.isnan(pp)
    err = (lambda a: np.mean(np.abs(y[ok] - a[ok]))) if cfg.SKILL_METRIC == "mae" else \
          (lambda a: np.sqrt(np.mean((y[ok] - a[ok]) ** 2)))
    return dict(mae=np.mean(np.abs(y - p)), rmse=np.sqrt(np.mean((y - p) ** 2)),
                skill=1 - err(p) / min(err(pc), err(pp)))


def reliability(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return pd.DataFrame([dict(bin=b, mean_pred=p[idx == b].mean(), observed=y[idx == b].mean(), n=int((idx == b).sum()))
                         for b in range(bins) if (idx == b).any()])