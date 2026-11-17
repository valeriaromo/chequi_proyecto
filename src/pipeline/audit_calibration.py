import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from common import OUT, HORIZON, FOLDS, load_master, rain_target, feature_cols, train_mask

H = HORIZON
EPS = 1e-4
VARIANTS = ["raw", "raw_fitonly", "platt", "iso"]
# AJUSTAR: usar los mismos hiperparámetros que rain_leakage.py
PARAMS =dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def bins_table(p, y, bins=10):
    b = pd.qcut(p, bins, duplicates="drop")
    return (pd.DataFrame({"p": p, "y": y, "b": b}).groupby("b", observed=True)
            .agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size")))


def ece(p, y):
    t = bins_table(p, y)
    return float((t.n * (t.p - t.y).abs()).sum() / t.n.sum())


def cal_slope(p, y):
    if len(np.unique(y)) < 2:  # fold parcial sin ambas clases
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def climatology(train, ytr, test, k=15):
    """Frecuencia por municipio y día del año (entrenamiento), suavizada circularmente ±k días."""
    d = pd.DataFrame({"m": train.municipio.values, "doy": train.date.dt.dayofyear.clip(upper=365).values, "y": ytr})
    tab = d.groupby(["m", "doy"], observed=True)["y"].mean().unstack().reindex(columns=range(1, 366))
    tab = tab.T.interpolate(limit_direction="both").T
    v = tab.to_numpy()
    sm = uniform_filter1d(np.concatenate([v[:, -k:], v, v[:, :k]], axis=1), 2 * k + 1, axis=1, mode="nearest")[:, k:-k]
    doy = test.date.dt.dayofyear.clip(upper=365).to_numpy()
    return np.clip(sm[tab.index.get_indexer(test.municipio), doy - 1], EPS, 1 - EPS)


def metrics(g):
    y = g.y.to_numpy()
    bc = brier(g.p_clim.to_numpy(), y)
    rows = []
    for v in VARIANTS:
        p = g[v].to_numpy()
        s, i = cal_slope(p, y)
        rows.append(dict(variant=v, n=len(g), brier=brier(p, y), bss=1 - brier(p, y) / bc, ece=ece(p, y),
                         slope=s, intercept=i, mean_p=p.mean(), obs_rate=y.mean()))
    return pd.DataFrame(rows)


def bss_ci(o, col="raw", B=1000, seed=0):
    """IC 95% del BSS con bootstrap por bloques de mes (respeta autocorrelación; con 4 folds es lo mínimo honesto)."""
    rng = np.random.default_rng(seed)
    key = o.date.dt.to_period("M").astype(str).to_numpy()
    groups = [np.flatnonzero(key == k) for k in np.unique(key)]
    y, p, pc = o.y.to_numpy(), o[col].to_numpy(), o.p_clim.to_numpy()
    vals = []
    for _ in range(B):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        vals.append(1 - brier(p[idx], y[idx]) / brier(pc[idx], y[idx]))
    return np.percentile(vals, [2.5, 97.5])


df = load_master()
X = feature_cols(df)
y_all = rain_target(df, H)
parts = []
for Y in FOLDS:
    te = (df.year == Y) & y_all.notna()
    if te.sum() == 0:
        continue
    tr = train_mask(df, Y, H) & y_all.notna()         # todo lo anterior a Y, con purga
    fit = train_mask(df, Y - 1, H) & y_all.notna()    # modelo para recalibrar
    cal = tr & ~fit                                   # año Y-1 como conjunto de calibración
    m_full = XGBClassifier(**PARAMS).fit(df.loc[tr, X], y_all[tr].astype(int))
    m_fit = XGBClassifier(**PARAMS).fit(df.loc[fit, X], y_all[fit].astype(int))
    p_cal = m_fit.predict_proba(df.loc[cal, X])[:, 1]
    y_cal = y_all[cal].astype(int).to_numpy()
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p_cal).reshape(-1, 1), y_cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS).fit(p_cal, y_cal)
    p_fo = m_fit.predict_proba(df.loc[te, X])[:, 1]
    parts.append(pd.DataFrame({
        "date": df.loc[te, "date"].to_numpy(), "municipio": df.loc[te, "municipio"].to_numpy(), "year": Y,
        "y": y_all[te].astype(int).to_numpy(),
        "p_clim": climatology(df[tr], y_all[tr].to_numpy(), df[te]),
        "raw": m_full.predict_proba(df.loc[te, X])[:, 1],
        "raw_fitonly": p_fo,
        "platt": platt.predict_proba(logit(p_fo).reshape(-1, 1))[:, 1],
        "iso": iso.predict(p_fo)}))

oof = pd.concat(parts, ignore_index=True)
oof.to_csv(OUT / "oof_rain_h1.csv", index=False)

by_fold = pd.concat([metrics(g).assign(fold=Y) for Y, g in oof.groupby("year", observed=True)], ignore_index=True)
pooled = metrics(oof).assign(fold="pooled")
lo, hi = bss_ci(oof, "raw")
by_fold.to_csv(OUT / "calibration_by_fold.csv", index=False)
pooled.to_csv(OUT / "calibration_pooled.csv", index=False)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0, 1], [0, 1], "k--", lw=1)
for v in ["raw", "platt", "iso"]:
    t = bins_table(oof[v].to_numpy(), oof.y.to_numpy())
    ax.plot(t.p, t.y, "o-", label=v)
ax.set(xlabel="probabilidad predicha", ylabel="frecuencia observada", title="Reliability, walk-forward 2023-2026")
ax.legend()
fig.savefig(OUT / "reliability.png", dpi=150, bbox_inches="tight")

print(pooled.round(3).to_string(index=False))
print(f"\nBSS raw pooled IC95% (bloques de mes): [{lo:.3f}, {hi:.3f}]")
r = pooled.set_index("variant").loc["raw"]
print("\nDECISIÓN (umbrales orientativos):")
if lo <= 0:
    print("- El IC del BSS incluye 0: la habilidad de lluvia h=1 no está demostrada.")
if abs(r.slope - 1) > 0.2 or r.ece > 0.03:
    best = pooled[pooled.variant.isin(["platt", "iso"])].sort_values("ece").iloc[0].variant
    print(f"- Mal calibrado (slope={r.slope:.2f}, ECE={r.ece:.3f}): recalibrar con '{best}' antes de congelar.")
else:
    print("- Calibración aceptable: congelar el modelo raw.")