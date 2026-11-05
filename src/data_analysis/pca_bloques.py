from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import pca_analysis as pca
import test_pca as T

STATIC = ["elevacion_msnm", "distancia_rio_m", "geom_centroid_lat", "geom_centroid_lon"]
CAL = ["doy_sin", "doy_cos", "month"]
L = ("lag1", "lag2", "lag3", "lag7")
lags = lambda v: [f"{v}_{s}" for s in L]
means = lambda v: [f"{v}_mean7d", f"{v}_mean30d"]

BLOCKS = {
    "temp": ["T2M", "T2M_MAX", "T2M_MIN", "t2m_rango", *lags("T2M"), *lags("T2M_MAX"),
             *lags("T2M_MIN"), *means("T2M")],
    "lluvia": ["PRECTOTCORR", "precipitation_mm", "rain_3d", "rain_7d", "rain_30d", "rain_90d",
               "dias_sin_lluvia", *lags("PRECTOTCORR")],
    "atmos": ["RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN", "vpd_kpa",
              *[c for v in ("RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN") for c in (*lags(v), *means(v))]],
}
LOG1P = set(BLOCKS["lluvia"])
EXCLUDE = {"date", "municipio", "year", "qc_aod_imputed"}          # igual que train_xgboost.py
is_aod = lambda c: c.startswith("aod_") or c == "qc_aod_pocos_pixeles"


def master_features(path) -> list[str]:
    # features reales de producción (sin AOD), leídas del parquet y no de los bloques
    return [c for c in pq.read_schema(path).names if c not in EXCLUDE and not is_aod(c)]


def check_cols(feats: list[str]) -> None:
    used = sum(BLOCKS.values(), []) + STATIC + CAL
    dup = {c for c in used if used.count(c) > 1}
    assert not dup, f"columnas en más de un bloque: {sorted(dup)}"
    falta, sobra = sorted(set(feats) - set(used)), sorted(set(used) - set(feats))
    assert not falta and not sobra, f"sin asignar: {falta} | no existen en el master: {sobra}"
    print(f"check_cols OK: {len(feats)} features = {', '.join(f'{k} {len(v)}' for k, v in BLOCKS.items())}"
          f" + {len(STATIC)} estáticas + {len(CAL)} calendario")


def fit_block(df, cols, train):
    # todo se ajusta solo con `train`: climatología, z-score y PCA; test solo se transforma
    X = df[cols].astype(float)
    X = X.assign(**{c: np.log1p(X[c].clip(lower=0)) for c in cols if c in LOG1P})
    A = T.anomalies(df[["municipio", "date"]].join(X), cols, train)
    mu, sd = A[train].mean(), A[train].std(ddof=0).replace(0, 1.0)
    Z = ((A - mu) / sd).fillna(0.0)
    res = pca.fit_pca(Z[train])
    W = res.loadings.to_numpy() / np.sqrt(res.eigenvalues)
    scores = pd.DataFrame(Z.to_numpy() @ W, index=df.index, columns=res.loadings.columns)
    cum = np.cumsum(res.eigenvalues) / np.sum(res.eigenvalues)   # varianza acumulada en train
    return scores, cum


def leak_check(df, cols, train, s1) -> None:
    # perturbo los años de test (x5): los scores de train no deben cambiar
    d2 = df[["municipio", "date", "year", *cols]].copy()
    d2[cols] = d2[cols].astype("float64")      # evita el FutureWarning float32 -> float64
    d2.loc[~train, cols] *= 5
    s2, _ = fit_block(d2, cols, train)
    assert np.allclose(s1[train], s2[train]), "fuga: el train depende del test"


def main() -> None:
    path = T.default_master_path()
    check_cols(master_features(path))
    df, _ = T.load_data(path)
    df["year"] = df["date"].dt.year

    rows = []
    for yr in T.TEST_YEARS:
        train = df["year"] < yr
        for name, cols in BLOCKS.items():
            s1, cum = fit_block(df, cols, train)
            leak_check(df, cols, train, s1)
            rows.append(dict(fold=yr, bloque=name, p=len(cols),
                             k80=int(np.searchsorted(cum, 0.80) + 1),
                             k90=int(np.searchsorted(cum, 0.90) + 1),
                             pc1=round(float(cum[0]), 3)))
            print(f"fold {yr} | {name}: sin fugas")

    res = pd.DataFrame(rows)
    out = T.OUT_DIR / "pca_bloques"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "k_piso_folds.csv", index=False)
    print("\n" + res.to_string(index=False))
    piso = res.groupby("bloque")[["k80", "k90"]].max()
    print("\nPiso de k por bloque (máximo entre folds):")
    print(piso)
    piso.to_csv(out / "k_piso.csv")


if __name__ == "__main__":
    main()