import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MASTER_PATH = ROOT / "data" / "master" / "master_table.parquet"
OUT = ROOT / "outputs_fase2"
OUT.mkdir(exist_ok=True)

# se usan para evaluar la habilidad de los modelos
BASE_COLS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PS","ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "precipitation_mm"]
# se usan para evaluar la fuga de informacion (leakage) en el master
AOD_COLS = ["aod_055_mean", "aod_055_count", "aod_qa_best_pct", "aod_055_mean_7d","qc_aod_pocos_pixeles", "qc_aod_imputed"]
TARGETS = {"T2M_MIN": "tmin", "T2M_MAX": "tmax", "precipitation_mm": "rain"}
EXCLUDE = ["date", "municipio", "year", "qc_aod_imputed", "aod_055_mean","aod_055_count", "aod_qa_best_pct", "aod_055_mean_7d", "qc_aod_pocos_pixeles"]
FOLD_YEARS = [2023, 2024, 2025, 2026]
HORIZONS = [1, 3, 7] # evaluar a 1, 3 y 7 dias de horizonte
WET_MM = 1.0
XGB_KW = dict(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8,colsample_bytree=0.8, n_jobs=-1, random_state=42)


def load():
    df = pd.read_parquet(MASTER_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["municipio", "date"]).reset_index(drop=True)


def add_target(df, var, h): # agrega columna "y" con el valor de la variable objetivo a predecir h dias en el futuro
    d = df.copy()
    d["y"] = d.groupby("municipio")[var].shift(-h)
    d["tdoy"] = (d["date"] + pd.Timedelta(days=h)).dt.dayofyear
    return d.dropna(subset=["y"])


def feature_cols(df): 
    return [c for c in df.columns if c not in EXCLUDE + ["y", "tdoy"]
            and c not in TARGETS and not c.startswith("target_")]


def split(d, year, h): # divide en train/test por año y horizonte
    cut = pd.Timestamp(f"{year}-01-01")
    tr = d[d["date"] < cut - pd.Timedelta(days=h)]      # es una purga para que no haya leakage de features calculadas con datos futuros
    te = d[(d["date"] >= cut) & (d["date"] < pd.Timestamp(f"{year + 1}-01-01"))]
    return tr, te


def smooth_doy(s, win=15): # suaviza una serie de 1 a 366 (doy) con media movil centrada de ventana win
    s = s.reindex(range(1, 367))
    ext = pd.concat([s.iloc[-win:], s, s.iloc[:win]])
    out = ext.rolling(win, center=True, min_periods=1).mean()
    return out.iloc[win:-win].set_axis(range(1, 367))


def climatology(tr, te, stat="mean"): # calcula la climatologia de wet y de mu (media de wet) para el test
    pred = pd.Series(np.nan, index=te.index)
    for m, g in tr.groupby("municipio"):
        prof = smooth_doy(g.groupby("tdoy")["y"].agg(stat))
        idx = te.index[te["municipio"] == m]
        pred.loc[idx] = te.loc[idx, "tdoy"].map(prof).values
    return pred


def test_leakage(n_cutoffs=12, seed=0):
    from build_master_table import add_features as FEATURE_FN 
    df = load()
    keys = ["municipio", "date"]
    base = df[keys + [c for c in BASE_COLS + AOD_COLS if c in df.columns]]
    ref = FEATURE_FN(base.copy()).sort_values(keys).reset_index(drop=True)
    feats = [c for c in ref.columns if c not in keys + list(base.columns)]

    # aplicar la funcion de features al master debe reproducir exactamente las features del master
    chk = ref.merge(df[keys + feats], on=keys, suffixes=("", "_m"))
    bad = [c for c in feats if not np.allclose(chk[c], chk[c + "_m"],
                                               equal_nan=True, rtol=1e-6)]
    print("No reproducen el master:", bad or "ninguna")

    # simular fuga de informacion (leakage) en el master y verificar que la funcion de features detecta la fuga
    rng = np.random.default_rng(seed)
    dates = np.sort(df["date"].unique())
    cutoffs = rng.choice(dates[120:-10], n_cutoffs, replace=False)
    leaks = {}
    for t in cutoffs:
        t = pd.Timestamp(t)
        b2 = base.copy()
        fut = b2["date"] > t
        for c in [c for c in BASE_COLS + AOD_COLS if c in b2.columns]:
            col = b2[c].astype(float)
            b2.loc[fut, c] = col[fut] * rng.uniform(0.3, 3.0, fut.sum()) + 1000
        new = FEATURE_FN(b2).sort_values(keys).reset_index(drop=True)
        past = ref["date"] <= t
        for c in feats:
            if not np.allclose(ref.loc[past, c], new.loc[past, c],
                               equal_nan=True, rtol=1e-6):
                leaks.setdefault(c, []).append(str(t.date()))
    if leaks:
        print("LEAKAGE detectado en:")
        for c, ts in leaks.items():
            print(f"  {c}: {len(ts)}/{n_cutoffs} cortes (ej. {ts[0]})")
        sys.exit(1)
    print(f"OK: {len(feats)} features sin informacion futura en {n_cutoffs} cortes.")


def eval_rain(): # evalua la habilidad de los modelos de lluvia (clasificacion y regresion) frente a la climatologia y persistencia
    df = load()
    feats = feature_cols(df)
    rows = []
    for h in HORIZONS:
        d = add_target(df, "precipitation_mm", h)
        d["wet"] = (d["y"] >= WET_MM).astype(int)
        for yr in FOLD_YEARS:
            tr, te = split(d, yr, h)
            if len(te) == 0:
                continue
            p_clim = climatology(tr.assign(y=tr["wet"]), te).clip(1e-3, 1 - 1e-3)
            wet_tr = tr[tr["wet"] == 1]
            mu_clim = climatology(wet_tr, te).fillna(wet_tr["y"].mean())
            ev_clim = p_clim * mu_clim

            # dos etapas
            clf = XGBClassifier(**XGB_KW).fit(tr[feats], tr["wet"])
            reg = XGBRegressor(objective="reg:gamma", **XGB_KW).fit(
                wet_tr[feats], wet_tr["y"])
            p = clf.predict_proba(te[feats])[:, 1]
            mu = reg.predict(te[feats])
            ev_2s = p * mu

            # tweedie
            tw = XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5,
                              **XGB_KW).fit(tr[feats], tr["y"])
            ev_tw = tw.predict(te[feats])

            y = te["y"].values; w = te["wet"].values
            wetm = w == 1
            b_clim = brier_score_loss(w, p_clim)
            b_xgb = brier_score_loss(w, p)

            def dev(yt, yp, pw=1.5):
                yp = np.clip(yp, 1e-4, None)
                return np.mean(2 * (np.where(yt > 0, yt ** (2 - pw), 0) /
                               ((1 - pw) * (2 - pw)) - yt * yp ** (1 - pw) /
                               (1 - pw) + yp ** (2 - pw) / (2 - pw)))

            rows.append(dict(
                h=h, year=yr, wet_frac=w.mean(),
                brier_clim=b_clim, brier_xgb=b_xgb, BSS=1 - b_xgb / b_clim,
                auc_xgb=roc_auc_score(w, p),
                dev_clim=dev(y, ev_clim.values), dev_2etapas=dev(y, ev_2s),
                dev_tweedie=dev(y, ev_tw),
                mae_wet_clim=mean_absolute_error(y[wetm], mu_clim.values[wetm]),
                mae_wet_xgb=mean_absolute_error(y[wetm], mu[wetm]),
            ))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "rain_eval.csv", index=False)
    pd.set_option("display.width", 220)
    print(r.round(3).to_string(index=False))
    print("\nPromedio por horizonte:")
    print(r.groupby("h").mean(numeric_only=True).drop(columns="year").round(3))
    print("\nCriterio: BSS > 0 y dev_2etapas/dev_tweedie < dev_clim en la mayoria "
          "de años; si no, no hay habilidad en lluvia.")


def eval_scope(var="T2M_MAX"):  # cambiar var segun la variable objetivo a evaluar (T2M_MAX, T2M_MIN o precipitation_mm)
    # evalua la habilidad de los modelos de prediccion de temperatura maxima, 
    # minima y lluvia frente a la climatologia y persistencia
    df = load()
    feats = feature_cols(df)
    rows = []
    for h in HORIZONS:
        d = add_target(df, var, h)
        for yr in FOLD_YEARS:
            tr, te = split(d, yr, h)
            if len(te) == 0:
                continue
            clim = climatology(tr, te)
            pers = te[var]
            best = min(mean_absolute_error(te["y"], clim),
                       mean_absolute_error(te["y"], pers))

            # global
            g = XGBRegressor(**XGB_KW).fit(tr[feats], tr["y"])
            mae_g = mean_absolute_error(te["y"], g.predict(te[feats]))

            # por municipio
            pred = pd.Series(np.nan, index=te.index)
            for m, trm in tr.groupby("municipio"):
                idx = te.index[te["municipio"] == m]
                mm = XGBRegressor(**XGB_KW).fit(trm[feats], trm["y"])
                pred.loc[idx] = mm.predict(te.loc[idx, feats])
            mae_m = mean_absolute_error(te["y"], pred)

            rows.append(dict(target=var, h=h, year=yr, best_baseline=best,
                             mae_global=mae_g, mae_por_muni=mae_m,
                             skill_global=1 - mae_g / best,
                             skill_por_muni=1 - mae_m / best))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"scope_{var}.csv", index=False)
    print(r.round(3).to_string(index=False))
    print("\nPromedio por horizonte:")
    print(r.groupby("h")[["skill_global", "skill_por_muni"]].mean().round(3))
    print("\nElige por muni solo si gana en la mayoria de folds y h; ""empate => global (menos mantenimiento).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"leakage": test_leakage, "rain": eval_rain, "scope": eval_scope}.get( # orquestra los tests de leakage, lluvia y scope
        cmd, lambda: print(__doc__))()