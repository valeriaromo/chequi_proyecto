from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import pca_analysis as pca
from variable_analysis import HERE, default_master_path, load_data

OUT_DIR = HERE / "output" / "data_analysis"
TEST_YEARS = [2023, 2024, 2025, 2026]
TARGET = "T2M_MIN"
RAIN_COL = "precipitation_mm"      # columna con la que se define lluvia (>= RAIN_MM)
RAIN_MM = 1.0
TOL_MEAN, TOL_FOLD = 0.02, 0.05    # tolerancias de pérdida vs "orig"
WIN = 15
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,colsample_bytree=0.8, tree_method="hist", random_state=pca.SEED, n_jobs=-1)


def anomalies(df: pd.DataFrame, cols: list[str], train: pd.Series) -> pd.DataFrame:
    # anomalía respecto a la climatología suavizada (circular) por municipio y día del año, calculada solo con `train`
    doy = df["date"].dt.dayofyear.clip(upper=366).to_numpy()
    munis = pd.Index(sorted(df["municipio"].unique()))
    mrow = munis.get_indexer(df["municipio"])
    out = pd.DataFrame(index=df.index)
    tr = df[train]
    for c in cols:
        prof = (tr.groupby(["municipio", tr["date"].dt.dayofyear.clip(upper=366)])[c].mean().unstack()
                .reindex(index=munis, columns=range(1, 367)))
        ext = pd.concat([prof.iloc[:, -WIN:], prof, prof.iloc[:, :WIN]], axis=1)
        sm = ext.T.rolling(WIN, center=True, min_periods=1).mean().T.iloc[:, WIN:-WIN]
        out[c] = df[c].to_numpy() - sm.to_numpy()[mrow, doy - 1]
    return out


def fold_data(df: pd.DataFrame, vars_: list[str], test_year: int):
    train = df["year"] < test_year
    test = df["year"] == test_year
    A = anomalies(df, vars_, train)
    mu, sd = A[train].mean(), A[train].std(ddof=0)
    Z = ((A - mu) / sd).fillna(0.0)    # NaN (AOD) -> 0 tras el z-score, como en pattern_discovery
    res = pca.fit_pca(Z[train])
    W = res.loadings.to_numpy() / np.sqrt(res.eigenvalues)   # vectores propios (p x p)
    scores = pd.DataFrame(Z.to_numpy() @ W, index=df.index, columns=res.loadings.columns)
    k80 = pca.components_needed(res)

    g = df.groupby("municipio")
    y_t = A[TARGET].groupby(df["municipio"]).shift(-1)    # anomalía de T2M_MIN en t+1
    y_r = (g[RAIN_COL + "_raw"].shift(-1) >= RAIN_MM).astype(float).where(g[RAIN_COL + "_raw"].shift(-1).notna())
    freq = (pd.DataFrame({"m": df["municipio"], "mo": df["date"].dt.month, "y": y_r})[train & y_r.notna()]
            .groupby(["m", "mo"])["y"].mean())
    keys = pd.MultiIndex.from_arrays([df["municipio"], df["date"].dt.month])
    p_clim = pd.Series(freq.reindex(keys).to_numpy(), index=df.index).fillna(y_r[train].mean())
    return Z, scores, y_t, y_r, p_clim, train, test, k80


def evaluate(X, y_t, y_r, p_clim, train, test) -> tuple[float, float]:
    mt_tr, mt_te = train & y_t.notna(), test & y_t.notna()
    reg = XGBRegressor(**XGB).fit(X[mt_tr], y_t[mt_tr])
    mse = np.mean((reg.predict(X[mt_te]) - y_t[mt_te]) ** 2)
    skill = 1 - mse / np.mean(y_t[mt_te] ** 2)          # baseline: anomalía 0 = climatología

    mr_tr, mr_te = train & y_r.notna(), test & y_r.notna()
    clf = XGBClassifier(**XGB, eval_metric="logloss").fit(X[mr_tr], y_r[mr_tr])
    brier = np.mean((clf.predict_proba(X[mr_te])[:, 1] - y_r[mr_te]) ** 2)
    bss = 1 - brier / np.mean((p_clim[mr_te] - y_r[mr_te]) ** 2)
    return float(skill), float(bss)


def test_profile(profile: str, df0: pd.DataFrame, out_root: Path) -> None:
    pca.set_profile(profile)
    vars_ = list(pca.PCA_VARS)
    df = df0.copy()
    df["year"] = df["date"].dt.year
    df[RAIN_COL + "_raw"] = df[RAIN_COL]                      # lluvia cruda para el target
    for c in pca.LOG1P[profile]:
        df[c] = np.log1p(df[c].clip(lower=0))
    df = df.sort_values(["municipio", "date"]).reset_index(drop=True)

    rows = []
    for yr in TEST_YEARS:
        Z, scores, y_t, y_r, p_clim, train, test, k80 = fold_data(df, vars_, yr)
        variants = {"orig": Z} | {k: scores.iloc[:, :k] for k in range(1, len(vars_) + 1)}
        for k, X in variants.items():
            skill, bss = evaluate(X, y_t, y_r, p_clim, train, test)
            rows.append(dict(profile=profile, test_year=yr, k=k, k80_fold=k80, skill_T2M_MIN=skill, BSS_rain=bss))
        print(f"[{profile}] fold {yr}: k80 = {k80}")
    res = pd.DataFrame(rows)

    summ = res.groupby("k", sort=False).agg(skill_mean=("skill_T2M_MIN", "mean"), skill_min=("skill_T2M_MIN", "min"),
                                           bss_mean=("BSS_rain", "mean"), bss_min=("BSS_rain", "min")).reset_index()
    ref = summ[summ.k == "orig"].iloc[0]
    ref_folds = res[res.k == "orig"].set_index("test_year")
    ok = []
    for k in range(1, len(vars_) + 1):
        s = summ[summ.k == k].iloc[0]
        f = res[res.k == k].set_index("test_year")
        good = (s.skill_mean >= ref.skill_mean - TOL_MEAN and s.bss_mean >= ref.bss_mean - TOL_MEAN
                and (f.skill_T2M_MIN >= ref_folds.skill_T2M_MIN - TOL_FOLD).all()
                and (f.BSS_rain >= ref_folds.BSS_rain - TOL_FOLD).all())
        ok.append(good)
    summ["cumple_criterio"] = ["ref"] + ok           # summ viene en el orden: orig, 1..p
    k_star = next((k for k, g in zip(range(1, len(vars_) + 1), ok) if g), None)

    tab = out_root / profile / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    res.to_csv(tab / "k_test_folds.csv", index=False)
    summ.to_csv(tab / "k_test_summary.csv", index=False)
    print(f"\n[{profile}] k80 por fold: {res.groupby('test_year').k80_fold.first().to_dict()}")
    print(summ.round(3).to_string(index=False))
    print(f"[{profile}] k recomendado: {k_star if k_star else 'ninguno cumple: usar las variables originales'}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", choices=[*pca.PROFILES, "ambos"], default="sin_aod")
    a = ap.parse_args()
    df0, _ = load_data(Path(a.data) if a.data else default_master_path())
    for p in (list(pca.PROFILES) if a.profile == "ambos" else [a.profile]):
        test_profile(p, df0, Path(a.out) if a.out else OUT_DIR)


if __name__ == "__main__":
    main()