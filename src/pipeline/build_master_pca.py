"""
Cuidado con el uso: con el ajuste por defecto el PCA ve TODOS los años, asi que esta tabla sirve para entrenar el
modelo final y servir en produccion. Un backtest con ella filtra informacion del futuro en las features; para
backtests usa --fit-until (un solo corte) o los PCs por fold de src/data_analysis/pca_walkforward.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import platform
import sys
from datetime import date
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MASTER_DIR, MASTER_TABLE_PATH, PARQUET_COMPRESSION
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

STATIC = ["elevacion_msnm", "distancia_rio_m", "geom_centroid_lat", "geom_centroid_lon"]
CAL = ["doy_sin", "doy_cos", "month"]
_L = ("lag1", "lag2", "lag3", "lag7")
lags = lambda v: [f"{v}_{s}" for s in _L]
means = lambda v: [f"{v}_mean7d", f"{v}_mean30d"]

BLOCKS = {
    "temp": ["T2M", "T2M_MAX", "T2M_MIN", "t2m_rango", *lags("T2M"), *lags("T2M_MAX"),
             *lags("T2M_MIN"), *means("T2M")],
    "lluvia": ["PRECTOTCORR", "precipitation_mm", "rain_3d", "rain_7d", "rain_30d", "rain_90d",
               "dias_sin_lluvia", *lags("PRECTOTCORR")],
    "atmos": ["RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN", "vpd_kpa",
              *[c for v in ("RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN") for c in (*lags(v), *means(v))]],
}
LOG1P = set(BLOCKS["lluvia"])                      # lluvia y sus derivadas pasan por log1p antes de la anomalia
TEMP_TARGETS = ("T2M_MAX", "T2M_MIN")
EXCLUDE = {"date", "municipio", "year", "qc_aod_imputed"}          # igual que train_xgboost.py
is_aod = lambda c: c.startswith("aod_") or c == "qc_aod_pocos_pixeles"
CLIM_WIN = 15                                      # suavizado circular de la climatologia (igual que test_pca.WIN)
LEVELS = (0.80, 0.90, 0.95)
DEFAULT_PCA_PATH = MASTER_DIR / "master_table_pca.parquet"


# features del master y bloques
def master_features(path: Path) -> list[str]:
    # las features que usa produccion sin AOD, leidas del parquet y no de los bloques
    return [c for c in pq.read_schema(path).names if c not in EXCLUDE and not is_aod(c)]


def check_blocks(feats: list[str]) -> None:
    used = sum(BLOCKS.values(), []) + STATIC + CAL
    dup = {c for c in used if used.count(c) > 1}
    assert not dup, f"columnas en mas de un bloque: {sorted(dup)}"
    falta, sobra = sorted(set(feats) - set(used)), sorted(set(used) - set(feats))
    assert not falta and not sobra, f"sin asignar: {falta} | no existen en el master: {sobra}"


# climatologia, z-score y PCA congelados
def clim_profile(F: pd.DataFrame, col: str, mask: pd.Series, munis: pd.Index) -> np.ndarray:
    # climatologia suavizada (circular) por municipio y dia del año, solo con las filas de `mask` -> (municipios x 366)
    tr = F[mask]
    prof = (tr.groupby(["municipio", tr["date"].dt.dayofyear.clip(upper=366)],observed=False)[col].mean().unstack().reindex(index=munis, columns=range(1, 367)))
    ext = pd.concat([prof.iloc[:, -CLIM_WIN:], prof, prof.iloc[:, :CLIM_WIN]], axis=1)
    return ext.T.rolling(CLIM_WIN, center=True, min_periods=1).mean().T.iloc[:, CLIM_WIN:-CLIM_WIN].to_numpy()


def clim_value(F: pd.DataFrame, prof: np.ndarray, munis: pd.Index, shift_days: int = 0) -> np.ndarray:
    doy = (F["date"] + pd.Timedelta(days=shift_days)).dt.dayofyear.clip(upper=366).to_numpy()
    mrow = munis.get_indexer(F["municipio"])
    assert (mrow >= 0).all(), "municipio desconocido: el PCA congelado no lo conoce"
    return prof[mrow, doy - 1]


def prep_block(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float)
    for c in cols:
        if c in LOG1P:
            X[c] = np.log1p(X[c].clip(lower=0))
    return df[["municipio", "date"]].join(X)


def freeze_block(df: pd.DataFrame, cols: list[str], mask: pd.Series) -> dict:
    munis = pd.Index(sorted(df["municipio"].unique()))
    F = prep_block(df, cols)
    profs = {c: clim_profile(F, c, mask, munis) for c in cols}
    A = pd.DataFrame({c: F[c].to_numpy() - clim_value(F, profs[c], munis) for c in cols}, index=df.index)
    mu, sd = A[mask].mean(), A[mask].std(ddof=0).replace(0, 1.0)
    Z = ((A - mu) / sd).fillna(0.0)                      # NaN (lags iniciales) -> 0 tras el z-score
    _, S, Vt = np.linalg.svd(Z[mask].to_numpy(float), full_matrices=False)
    Vt = Vt * np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])[:, None]   # signos reproducibles
    eig = S ** 2 / (int(mask.sum()) - 1)
    return dict(cols=list(cols), munis=list(munis), profs=profs, mu=mu, sd=sd, W=Vt.T,
                names=[f"PC{i + 1}" for i in range(len(S))], eigenvalues=eig, cum=np.cumsum(eig) / eig.sum())


def transform_block(df: pd.DataFrame, b: dict) -> pd.DataFrame:
    # solo transforma: usa los objetos congelados, nada se ajusta con las filas nuevas
    munis = pd.Index(b["munis"])
    F = prep_block(df, b["cols"])
    A = pd.DataFrame({c: F[c].to_numpy() - clim_value(F, b["profs"][c], munis) for c in b["cols"]}, index=df.index)
    Z = ((A - b["mu"]) / b["sd"]).fillna(0.0)
    return pd.DataFrame(Z.to_numpy() @ b["W"], index=df.index, columns=b["names"])


def k_for(b: dict, level: float) -> int:
    return int(np.searchsorted(b["cum"], level) + 1)


def freeze_all(df: pd.DataFrame, feats: list[str], mask: pd.Series) -> dict:
    munis = pd.Index(sorted(df["municipio"].unique()))
    F = df[["municipio", "date", *TEMP_TARGETS]]
    return dict(blocks={n: freeze_block(df, c, mask) for n, c in BLOCKS.items()}, munis=list(munis),
                target_clim={t: clim_profile(F, t, mask, munis) for t in TEMP_TARGETS},   # para modelos de anomalia
                feats=list(feats), fecha_ajuste_ini=str(df.loc[mask, "date"].min().date()),
                fecha_ajuste_fin=str(df.loc[mask, "date"].max().date()), n_filas_ajuste=int(mask.sum()))


def pc_frame(df: pd.DataFrame, art: dict, level: float) -> pd.DataFrame:
    parts = []
    for n, b in art["blocks"].items():
        S = transform_block(df, b).iloc[:, :k_for(b, level)]
        S.columns = [f"{n}_{c}" for c in S.columns]       # temp_PC1, lluvia_PC1, atmos_PC1, ...
        parts.append(S)
    return pd.concat(parts, axis=1)


# ---------- manifest, guardado y verificacion ----------
def build_manifest(art, df, feats, version, art_dir, db_path, master_path, fit_until, max_level, pc_cols) -> dict:
    levels = [lv for lv in LEVELS if lv <= max_level + 1e-9]
    pcs_lv = {str(lv): [f"{n}_{c}" for n, b in art["blocks"].items() for c in b["names"][:k_for(b, lv)]] for lv in levels}
    return dict(
        version=version, creado=str(date.today()), master=master_path.name,
        master_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest(),
        db=db_path.name, artefactos=str(art_dir.relative_to(MASTER_DIR)) if art_dir.is_relative_to(MASTER_DIR) else str(art_dir),
        fit_until=fit_until, ajuste=dict(ini=art["fecha_ajuste_ini"], fin=art["fecha_ajuste_fin"], filas=art["n_filas_ajuste"]),
        n_filas=len(df), fecha_ini=str(df["date"].min().date()), fecha_fin=str(df["date"].max().date()),
        bloques={n: dict(p=len(b["cols"]), features=b["cols"], k_por_nivel={str(lv): k_for(b, lv) for lv in levels},
                         varianza_acumulada=[round(float(x), 4) for x in b["cum"]]) for n, b in art["blocks"].items()},
        pc_cols=pc_cols, pcs_por_nivel=pcs_lv, originales=feats, estaticas=STATIC, calendario=CAL,
        representaciones={"orig65": feats, **{f"solo_pcs_{lv}": pcs_lv[str(lv)] + STATIC + CAL for lv in levels},
                          **{f"orig+pcs_{lv}": feats + pcs_lv[str(lv)] for lv in levels}},
        libs=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__))


def verify(df: pd.DataFrame, db_path: Path, art_dir: Path, pc_cols: list[str], level: float) -> None:
    art = joblib.load(art_dir / "pca_bloques.joblib")             # lo que leeria produccion
    out = pd.read_parquet(db_path)
    assert len(out) == len(df) and not out.duplicated(["municipio", "date"]).any(), "filas/llaves distintas al master"
    assert set(df.columns) <= set(out.columns), "faltan columnas del master en la tabla con PCs"
    assert not out[pc_cols].isna().any().any(), "PCs con nulos"
    again = pc_frame(df, art, level)                               # transformar de nuevo con el objeto recargado
    assert list(again.columns) == pc_cols
    assert np.allclose(out[pc_cols].to_numpy(float), again.to_numpy(), rtol=1e-4, atol=1e-4), "PCs guardados != transformacion congelada"
    for n, b in art["blocks"].items():
        assert b["cum"][k_for(b, level) - 1] >= level - 1e-9, f"varianza del bloque {n} menor a {level}"
    logger.info(f"verificacion OK: {len(out)} filas, {len(pc_cols)} PCs, PCs == transformacion con objetos recargados")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help=f"master_table (default: {MASTER_TABLE_PATH})")
    ap.add_argument("--out-dir", default=None, help="carpeta de salida (default: data/master)")
    ap.add_argument("--max-level", type=float, default=0.95, help="varianza acumulada minima por bloque a guardar")
    ap.add_argument("--fit-until", default=None, help="AAAA-MM-DD: ajusta el PCA solo con filas hasta esa fecha")
    a = ap.parse_args()

    master_path = Path(a.data) if a.data else MASTER_TABLE_PATH
    out_root = Path(a.out_dir) if a.out_dir else MASTER_DIR
    feats = master_features(master_path)
    check_blocks(feats)
    df = pd.read_parquet(master_path)
    df["date"] = pd.to_datetime(df["date"])
    assert not df.duplicated(["municipio", "date"]).any(), "master con duplicados (municipio, date)"

    mask = pd.Series(True, index=df.index) if a.fit_until is None else df["date"] <= pd.Timestamp(a.fit_until)
    assert mask.sum() > 0, "--fit-until deja sin filas de ajuste"
    if a.fit_until is None:
        logger.warning("PCA ajustado con TODOS los años: no uses esta tabla para backtests (ver --fit-until)")
    art = freeze_all(df, feats, mask)

    suffix = "" if a.fit_until is None else f"_fit{pd.Timestamp(a.fit_until):%Y%m%d}"
    version = f"v{df['date'].max():%Y%m%d}_{hashlib.sha256(master_path.read_bytes()).hexdigest()[:8]}{suffix}"
    art_dir = out_root / "pca" / version
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(art, art_dir / "pca_bloques.joblib")

    pcs = pc_frame(df, art, a.max_level)
    pc_cols = list(pcs.columns)
    db_path = out_root / f"master_table_pca{suffix}.parquet"
    save_parquet(pd.concat([df, pcs], axis=1), db_path, partition_cols=None, compression=PARQUET_COMPRESSION)

    manifest = build_manifest(art, df, feats, version, art_dir, db_path, master_path,
                              a.fit_until, a.max_level, pc_cols)
    man_path = db_path.with_suffix(".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (art_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    for n, info in manifest["bloques"].items():
        logger.info(f"bloque {n}: {info['p']} features -> k por nivel {info['k_por_nivel']}")
    logger.info(f"{db_path.name}: {len(df)} filas, {df.shape[1]} columnas del master + {len(pc_cols)} PCs -> {db_path}")
    logger.info(f"objetos congelados -> {art_dir}")
    verify(df, db_path, art_dir, pc_cols, a.max_level)


if __name__ == "__main__":
    main()