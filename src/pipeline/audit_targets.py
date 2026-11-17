
import sys
import numpy as np
import pandas as pd
from common import (OUT, RAIN_COL, RAIN_THRESH, TMIN_COL, HORIZON, FOLDS,load_master, rain_target, tmin_target, feature_cols)

df = load_master()
H = HORIZON
res = []


def check(name, ok, detail=""):
    res.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})
    print(("PASS" if ok else "FAIL"), "|", name, "|", detail)


def info(name, detail):
    res.append({"check": name, "status": "INFO", "detail": detail})
    print("INFO |", name, "|", detail)


# 1. Grid diario consecutivo: shift(-h) equivale a "fecha + h días" solo si no hay huecos ni duplicados.
saltos = df.groupby("municipio", observed=True)["date"].apply(lambda s: int((s.diff().dropna() != pd.Timedelta(days=1)).sum()))
check("grid diario sin huecos ni duplicados", saltos.sum() == 0, f"saltos={int(saltos.sum())}")

# 2. Reconstrucción independiente: merge por (municipio, fecha + h) vs. shift(-h).
for name, col, fn in [("rain", RAIN_COL, rain_target), ("tmin", TMIN_COL, tmin_target)]:
    fut = df[["municipio", "date", col]].copy()
    fut["date"] = fut["date"] - pd.Timedelta(days=H)
    ref = df[["municipio", "date"]].merge(fut, on=["municipio", "date"], how="left")[col]
    if name == "rain":
        ref = (ref >= RAIN_THRESH).astype(float).where(ref.notna())
    tgt = fn(df, H)
    same = np.array_equal(ref.fillna(-999).to_numpy(), tgt.fillna(-999).to_numpy())
    check(f"target {name}: shift(-h) == merge en fecha+h", same)

# 3. Sin contaminación entre municipios: las últimas h filas de cada uno no tienen target.
tail = df.groupby("municipio", observed=True).tail(H).index
check("últimas h filas por municipio sin target",
      rain_target(df, H).loc[tail].isna().all() and tmin_target(df, H).loc[tail].isna().all())

# 4. Ninguna feature es copia del target ni del futuro de la variable fuente (p. ej. lag con signo invertido).
feats = feature_cols(df)
targets = {"rain": rain_target(df, H), "tmin": tmin_target(df, H)}
sospechas = []
for name, t in targets.items():
    ok = t.notna()
    corr = df.loc[ok, feats].corrwith(t[ok]).abs()
    sospechas += [(name, c, round(v, 3), "corr con target > 0.9") for c, v in corr[corr > 0.9].items()]
for col in (RAIN_COL, TMIN_COL):
    for k in range(1, 8):
        fk = df.groupby("municipio", observed=True)[col].shift(-k)
        ok = fk.notna()
        corr = df.loc[ok, feats].corrwith(fk[ok]).abs()
        sospechas += [(f"{col}(t+{k})", c, round(v, 4), "copia del futuro") for c, v in corr[corr > 0.999].items()]
check("sin features copia del target/futuro", len(sospechas) == 0, str(sospechas[:10]))

# 5. Solapamiento train/test por horizonte (informativo): filas de entrenamiento cuyo target cae en el año de test.
for Y in FOLDS:
    cutoff = pd.Timestamp(year=Y, month=1, day=1)
    n = int(((df["year"] < Y) & (df["date"] + pd.Timedelta(days=H) >= cutoff)).sum())
    info(f"fold {Y}: filas de train con target en año de test", f"{n} -> excluir con common.train_mask")

pd.DataFrame(res).to_csv(OUT / "audit_targets.csv", index=False)
sys.exit(0 if all(r["status"] != "FAIL" for r in res) else 1)