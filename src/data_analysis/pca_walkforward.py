"""Walk-forward semestral para decidir, por target y horizonte, si los PCs pueden reemplazar al master.

Cada fold es un semestre de test. Con solo las filas de train (fecha < inicio - h dias, igual que
train_xgboost.py) se ajustan climatologia, z-score, PCA y XGBoost; el test solo se transforma.
No toca pattern_discovery.py.

Uso (desde src/data_analysis):
    python pca_walkforward.py
    python pca_walkforward.py --horizons 1 --targets T2M_MIN lluvia      # corrida parcial
    python pca_walkforward.py --n-estimators 30                          # prueba rapida
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import test_pca as T
from pca_bloques import BLOCKS, STATIC, CAL, check_cols, fit_block, master_features

TARGETS = ("T2M_MAX", "T2M_MIN", "lluvia")
HORIZONS = (1, 3, 7)
LEVELS = (0.80, 0.90, 0.95)       # varianza acumulada minima por bloque, medida en el train de cada fold
FIRST_YEAR = 2023
MIN_TEST_DAYS = 90                # no se evalua un semestre con menos dias (2026-H2 solo tiene julio)
N_BOOT, SEED = 2000, 0


def make_folds(df: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    last, folds = df["date"].max(), []
    for yr in range(FIRST_YEAR, last.year + 1):
        for half, month in ((1, 1), (2, 7)):
            start = pd.Timestamp(yr, month, 1)
            end = start + pd.DateOffset(months=6)
            if (min(end, last + pd.Timedelta(days=1)) - start).days >= MIN_TEST_DAYS:
                folds.append((f"{yr}H{half}", start, end))
    return folds


def temp_target(df, tgt, h, train):
    # anomalia de T2M_MAX / T2M_MIN en t+h; climatologia suavizada por municipio y dia del año, solo con train
    A = T.anomalies(df[["municipio", "date", tgt]], [tgt], train)[tgt]
    return A.groupby(df["municipio"]).shift(-h)


def rain_target(df, h, train):
    # lluvia (>= 1 mm) en t+h y probabilidad climatologica por municipio y mes del dia objetivo, solo con train
    raw = df.groupby("municipio")[T.RAIN_COL].shift(-h)
    w = (raw >= T.RAIN_MM).astype(float).where(raw.notna())
    tm = (df["date"] + pd.Timedelta(days=h)).dt.month
    freq = (pd.DataFrame({"m": df["municipio"], "mo": tm, "y": w})[train & w.notna()]
            .groupby(["m", "mo"])["y"].mean())
    keys = pd.MultiIndex.from_arrays([df["municipio"], tm])
    p_clim = pd.Series(freq.reindex(keys).to_numpy(), index=df.index).fillna(w[train].mean())
    return w, p_clim


def eval_temp(X, y, train, test) -> float:
    tr, te = train & y.notna(), test & y.notna()
    reg = XGBRegressor(**T.XGB).fit(X[tr], y[tr])
    mse = np.mean((reg.predict(X[te]) - y[te]) ** 2)
    return float(1 - mse / np.mean(y[te] ** 2))              # baseline: anomalia 0 = climatologia


def eval_rain(X, w, p_clim, train, test) -> float:
    tr, te = train & w.notna(), test & w.notna()
    clf = XGBClassifier(**T.XGB, eval_metric="logloss").fit(X[tr], w[tr])
    brier = np.mean((clf.predict_proba(X[te])[:, 1] - w[te]) ** 2)
    return float(1 - brier / np.mean((p_clim[te] - w[te]) ** 2))    # BSS contra la climatologia


def boot_ci(d: np.ndarray) -> tuple[float, float]:
    # intervalo bootstrap pareado de la diferencia media, remuestreando folds (con pocos folds es amplio)
    rng = np.random.default_rng(SEED)
    m = rng.choice(d, size=(N_BOOT, len(d))).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run(df, feats, targets, horizons, levels):
    elig = sum(BLOCKS.values(), [])
    fixed = df[STATIC + CAL]
    rows, ks = [], []
    folds = make_folds(df)
    print(f"{len(folds)} folds: {[f[0] for f in folds]}")
    for h in horizons:
        for name, start, end in folds:
            train = df["date"] < start - pd.Timedelta(days=h)      # purga: y_{t+h} no toca el test
            test = (df["date"] >= start) & (df["date"] < end)
            fits = {b: fit_block(df, c, train) for b, c in BLOCKS.items()}   # una vez por (fold, h)
            variants = [("orig65", "-", df[feats])]
            for lv in levels:
                k = {b: int(np.searchsorted(cum, lv) + 1) for b, (_, cum) in fits.items()}
                ks.append(dict(h=h, fold=name, nivel=lv, **k))
                P = pd.concat([s.iloc[:, :k[b]].add_prefix(f"{b}_") for b, (s, _) in fits.items()], axis=1)
                variants.append(("solo_pcs", lv, pd.concat([P, fixed], axis=1)))
                variants.append(("orig+pcs", lv, pd.concat([df[feats], P], axis=1)))
            for tgt in targets:
                if tgt == "lluvia":
                    w, p_clim = rain_target(df, h, train)
                    score = lambda X: eval_rain(X, w, p_clim, train, test)
                else:
                    y = temp_target(df, tgt, h, train)
                    score = lambda X, y=y: eval_temp(X, y, train, test)
                for vname, lv, X in variants:
                    rows.append(dict(target=tgt, h=h, fold=name, variante=vname, nivel=lv,
                                     n_feats=X.shape[1], skill=score(X)))
                print(f"h={h} fold {name} {tgt}: ok")
    return pd.DataFrame(rows), pd.DataFrame(ks)


def summarize(res: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (tgt, h), g in res.groupby(["target", "h"]):
        ref = g[g.variante == "orig65"].set_index("fold")["skill"]
        rows.append(dict(target=tgt, h=h, variante="orig65", nivel="-", n_feats=int(g[g.variante == "orig65"].n_feats.iloc[0]),
                         skill_media=ref.mean(), skill_min=ref.min(), dif_media=0.0, dif_min=0.0, folds_mejor=0,
                         ic95_lo=0.0, ic95_hi=0.0, cumple=True))
        for (v, lv), gg in g[g.variante != "orig65"].groupby(["variante", "nivel"], sort=False):
            s = gg.set_index("fold")["skill"]
            d = (s - ref).to_numpy()
            lo, hi = boot_ci(d)
            rows.append(dict(target=tgt, h=h, variante=v, nivel=lv, n_feats=int(round(gg.n_feats.mean())),
                             skill_media=s.mean(), skill_min=s.min(), dif_media=d.mean(), dif_min=d.min(),
                             folds_mejor=int((d > 0).sum()), ic95_lo=lo, ic95_hi=hi,
                             cumple=bool(d.mean() >= -T.TOL_MEAN and (d >= -T.TOL_FOLD).all())))
    summ = pd.DataFrame(rows)
    # eleccion por target y horizonte: la variante que cumple con menos features (empate -> mayor dif_media)
    pick = []
    for (tgt, h), g in summ.groupby(["target", "h"]):
        ok = g[g.cumple].sort_values(["n_feats", "dif_media"], ascending=[True, False]).iloc[0]
        pick.append(ok)
    return summ, pd.DataFrame(pick).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS), choices=TARGETS)
    ap.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    ap.add_argument("--levels", nargs="+", type=float, default=list(LEVELS))
    ap.add_argument("--n-estimators", type=int, default=None, help="solo para pruebas rapidas")
    a = ap.parse_args()
    if a.n_estimators:
        T.XGB = {**T.XGB, "n_estimators": a.n_estimators}

    path = Path(a.data) if a.data else T.default_master_path()
    feats = master_features(path)
    check_cols(feats)
    df, _ = T.load_data(path)
    df["year"] = df["date"].dt.year

    res, ks = run(df, feats, a.targets, a.horizons, a.levels)
    summ, pick = summarize(res)
    out = (Path(a.out) if a.out else T.OUT_DIR) / "pca_walkforward"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "folds.csv", index=False)
    ks.to_csv(out / "k_por_fold.csv", index=False)
    summ.to_csv(out / "resumen.csv", index=False)
    pick.to_csv(out / "elegidas.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== Resumen por target y horizonte (dif = variante - orig65) ===")
    print(summ.round(3).to_string(index=False))
    print("\n=== Eleccion: variante que cumple con menos features ===")
    print(pick[["target", "h", "variante", "nivel", "n_feats", "dif_media", "dif_min", "folds_mejor",
                "ic95_lo", "ic95_hi"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()