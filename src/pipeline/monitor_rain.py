"""
Fase 3: congelar modelos finales y monitorear skill con regla de retiro.

Uso:
    python fase3_produccion.py train [FECHA_CORTE]   # ej. 2025-06-30 (ensayo)
    python fase3_produccion.py monitor [VENTANA]     # default 90 dias

Productos: T2M_MIN global h=1 y P(lluvia >= WET_MM) h=1. Nada mas.
Requiere rain_leakage.py en la misma carpeta.
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier, XGBRegressor
from rain_leakage import (load, add_target, feature_cols, smooth_doy,XGB_KW, WET_MM, BASE_COLS, OUT)

MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "modelo_v1.joblib"


def profiles(d, col):
    return {m: smooth_doy(g.groupby("tdoy")[col].mean()) for m, g in d.groupby("municipio")}


def apply_prof(prof, te):
    out = pd.Series(np.nan, index=te.index)
    for m, g in te.groupby("municipio"):
        out.loc[g.index] = g["tdoy"].map(prof[m]).values
    return out


def train(cutoff=None):
    df = load()
    if cutoff:
        df = df[df["date"] <= pd.Timestamp(cutoff)]
    feats = feature_cols(df)

    dt = add_target(df, "T2M_MIN", 1)
    reg = XGBRegressor(**XGB_KW).fit(dt[feats], dt["y"])

    dr = add_target(df, "precipitation_mm", 1)
    dr["wet"] = (dr["y"] >= WET_MM).astype(int)
    clf = XGBClassifier(**XGB_KW).fit(dr[feats], dr["wet"])

    art = dict(reg=reg, clf=clf, feats=feats,
               prof_t=profiles(dt, "y"), prof_w=profiles(dr, "wet"),
               trained_through=str(dt["date"].max().date()),
               n_train=len(dt), version="v1")
    joblib.dump(art, MODEL_PATH)
    print(f"Guardado {MODEL_PATH} | entrenado hasta {art['trained_through']} "
          f"| {len(dt)} filas | {len(feats)} features")


def rolling_skill(model_err, base_err, window):
    return 1 - model_err.rolling(window).mean() / base_err.rolling(window).mean()


def verdict(skill, window):
    s = skill.dropna()
    if s.empty:
        return "SIN_DATOS"
    now = s.iloc[-1]
    prev = s.iloc[-1 - window] if len(s) > window else np.nan
    if now < 0 and prev < 0:
        return "RETIRAR"
    if now < 0:
        return "ALERTA"
    return "OK"

def monitor(window=90):
    art = joblib.load(MODEL_PATH)
    feats = art["feats"]
    since = pd.Timestamp(art["trained_through"])
    df = load()
    status = dict(modelo=art["version"], entrenado_hasta=art["trained_through"],
                  ventana_dias=window)

    # --- T2M_MIN
    d = add_target(df, "T2M_MIN", 1)
    d = d[d["date"] > since].copy()
    if d["date"].nunique() >= window:
        d["pred"] = art["reg"].predict(d[feats])
        d["clim"] = apply_prof(art["prof_t"], d)
        e = pd.DataFrame({
            "m": (d["y"] - d["pred"]).abs(),
            "c": (d["y"] - d["clim"]).abs(),
            "p": (d["y"] - d["T2M_MIN"]).abs()}).groupby(d["date"]).mean()
        base = np.minimum(e["c"].rolling(window).mean(), e["p"].rolling(window).mean())
        sk = 1 - e["m"].rolling(window).mean() / base
        status["T2M_MIN"] = dict(skill_actual=float(sk.dropna().iloc[-1]),
                                 estado=verdict(sk, window))
    else:
        status["T2M_MIN"] = dict(estado="SIN_DATOS_SUFICIENTES",
                                 dias_disponibles=int(d["date"].nunique()))

    # --- Lluvia (Brier skill score)
    r = add_target(df, "precipitation_mm", 1)
    r = r[r["date"] > since].copy()
    if r["date"].nunique() >= window:
        r["wet"] = (r["y"] >= WET_MM).astype(int)
        p = art["clf"].predict_proba(r[feats])[:, 1]
        pc = apply_prof(art["prof_w"], r).clip(1e-3, 1 - 1e-3)
        b = pd.DataFrame({"m": (p - r["wet"]) ** 2,
                          "c": (pc - r["wet"]) ** 2}).groupby(r["date"]).mean()
        sk = 1 - b["m"].rolling(window).mean() / b["c"].rolling(window).mean()
        status["lluvia_prob"] = dict(BSS_actual=float(sk.dropna().iloc[-1]),
                                     estado=verdict(sk, window))
    else:
        status["lluvia_prob"] = dict(estado="SIN_DATOS_SUFICIENTES",
                                     dias_disponibles=int(r["date"].nunique()))

    # --- Salud de datos: NaN en columnas base, ultimos 30 dias
    last = df[df["date"] > df["date"].max() - pd.Timedelta(days=30)]
    nan_pct = last[[c for c in BASE_COLS if c in last.columns]].isna().mean()
    status["datos"] = dict(ultima_fecha=str(df["date"].max().date()),
                           nan_base_max_pct=float(nan_pct.max() * 100),
                           estado="ALERTA" if nan_pct.max() > 0.05 else "OK")

    (OUT / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "monitor":
        monitor(int(sys.argv[2]) if len(sys.argv) > 2 else 90)
    else:
        print(__doc__)