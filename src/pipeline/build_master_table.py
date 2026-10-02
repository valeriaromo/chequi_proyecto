from __future__ import annotations
import logging
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (FECHA_FIN_MASTER,FECHA_INICIO_MASTER,MASTER_DIR,MUNICIPIOS_AGS,PARQUET_COMPRESSION)
from src.utils.io_utils import save_parquet
from clean import FUENTES_DINAMICAS, FUENTES_ESTATICAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MUNICIPIOS_VALIDOS = list(MUNICIPIOS_AGS)
KEYS_DIN = ["date", "municipio"]
FUENTES_MASTER = [f for f in FUENTES_DINAMICAS if f != "openaq"] # OpenAQ (3.4% de celdas del estado de Aguascalientes) como feature equivale a la identidad municipal
ESTATICAS_MASTER = ["elevacion_msnm", "distancia_rio_m", "geom_centroid_lat", "geom_centroid_lon"]
ESTATICAS_DIM = [f for f in FUENTES_ESTATICAS if f != "geometria_municipios"]
RE_COLS_QC_RUIDO = re.compile(r"^qc_.*_(outlier|sospechoso|inconsistente)$")
LAG_VARS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN", "PRECTOTCORR"]
LAGS = (1, 2, 3, 7)
ROLL_VARS = ["T2M", "RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN"]
ROLL_WINDOWS = (7, 30)
RAIN_WINDOWS = (3, 7, 30, 90)
DIA_LLUVIOSO_MM = 1.0


def _read_master(nombre: str) -> pd.DataFrame:
    path = MASTER_DIR / f"{nombre}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe: corre clean.py antes del master")
    df = pd.read_parquet(path, engine="pyarrow").drop(columns=["source", "year"], errors="ignore")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    return df


def _features(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in keys and df[c].dtype.kind in "fiub" and not RE_COLS_QC_RUIDO.match(c)]
    out = df[keys + cols].copy()  # limpieza para XGBoost solo columnas numéricas y claves
    for c in cols:
        if out[c].dtype == bool:
            out[c] = out[c].astype("int8")
    return out


def _merge(master: pd.DataFrame, df: pd.DataFrame, on: list[str], nombre: str) -> pd.DataFrame:
    choque = (set(master.columns) & set(df.columns)) - set(on)
    if choque:
        raise ValueError(f"[{nombre}] columnas repetidas en el master: {sorted(choque)}")
    out = master.merge(df, on=on, how="left", validate="many_to_one")
    logger.info(f"Integrado {nombre}: {df.shape[1] - len(on)} columnas")
    return out


def build_canonical_grid() -> pd.DataFrame:
    dias = pd.date_range(FECHA_INICIO_MASTER, FECHA_FIN_MASTER, freq="D")
    grid = pd.MultiIndex.from_product([MUNICIPIOS_VALIDOS, dias], names=["municipio", "date"]).to_frame(index=False)
    logger.info(f"Grid canónico: {len(grid)} filas ({len(MUNICIPIOS_VALIDOS)} municipios x {len(dias)} días)")
    return grid


def _dry_streak(precip: pd.Series) -> pd.Series:
    wet = precip >= DIA_LLUVIOSO_MM  # variables helpers al XGBoost días consecutivos sin lluvia (0) y con lluvia (1)
    return (~wet).groupby(wet.cumsum()).cumsum()


def _roll(s, window: int, fn: str, min_periods: int | None = None) -> pd.Series:
    r = s.rolling(window, min_periods=window if min_periods is None else min_periods)
    return getattr(r, fn)().reset_index(level=0, drop=True)


def add_features(master: pd.DataFrame) -> pd.DataFrame:
    # se usan variables de lag y rolling para capturar la dinámica temporal de las series meteorológicas y de precipitación
    # ayuda a capturar la estacionalidad y tendencias a corto y mediano plazo, lo que puede mejorar la capacidad predictiva del modelo
    requeridas = set(LAG_VARS) | set(ROLL_VARS) | {"precipitation_mm", "T2M_MAX", "T2M_MIN", "aod_055_mean"}
    faltan = sorted(requeridas - set(master.columns))
    if faltan:
        raise ValueError(f"Faltan columnas para features derivadas: {faltan}")

   #proporcion = master["qc_aod_imputed"].mean() * 100   # verificar la proporción de AOD imputada (por nubosidad) para decidir si conviene usarla como feature
   #print(f"Proporción imputada: {proporcion:.2f}%")
    master = master.sort_values(["municipio", "date"]).reset_index(drop=True)
    g = master.groupby("municipio")
    new: dict[str, pd.Series] = {}
    for w in RAIN_WINDOWS:
        new[f"rain_{w}d"] = _roll(g["precipitation_mm"], w, "sum")
    new["dias_sin_lluvia"] = g["precipitation_mm"].transform(_dry_streak)
    for v in LAG_VARS:
        for lag in LAGS:
            new[f"{v}_lag{lag}"] = g[v].shift(lag)
    for v in ROLL_VARS:
        for w in ROLL_WINDOWS:
            new[f"{v}_mean{w}d"] = _roll(g[v], w, "mean")

    new["aod_055_mean_7d"] = _roll(g["aod_055_mean"], 7, "mean", min_periods=1)     # AOD tiene huecos por nubosidad: la media 7d usa solo los días observados
    new["t2m_rango"] = master["T2M_MAX"] - master["T2M_MIN"]
    es = 0.6108 * np.exp(17.27 * master["T2M"] / (master["T2M"] + 237.3))  # kPa
    new["vpd_kpa"] = es * (1 - master["RH2M"] / 100)
    doy = master["date"].dt.dayofyear
    new["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    new["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    new["month"] = master["date"].dt.month.astype("int8")
    new["year"] = master["date"].dt.year  # solo para el walk forward que se hace en el test no como feature del modelo
    return pd.concat([master, pd.DataFrame(new)], axis=1)

def build_dim_municipios() -> pd.DataFrame:
    dim = _read_master("geometria_municipios")
    for nombre in ESTATICAS_DIM:
        dim = _merge(dim, _read_master(nombre), ["municipio"], nombre)
    logger.info(f"dim_municipios: {len(dim)} filas, {dim.shape[1]} columnas")
    return dim


def build_master_table(dim: pd.DataFrame) -> pd.DataFrame:
    master = build_canonical_grid()
    for nombre in FUENTES_MASTER:
        master = _merge(master, _features(_read_master(nombre), KEYS_DIN), KEYS_DIN, nombre)

    estaticas = dim[["municipio", *ESTATICAS_MASTER]]
    if estaticas.isna().any().any():
        raise ValueError(f"Estáticas con nulos en dim_municipios:\n{estaticas[estaticas.isna().any(axis=1)]}")
    master = _merge(master, estaticas, ["municipio"], "estáticas")
    master = add_features(master)
    assert not master.duplicated(["municipio", "date"]).any(), "master con duplicados (municipio, date)"
    assert master["municipio"].nunique() == len(MUNICIPIOS_VALIDOS), "faltan municipios en el master"
    return master

def print_quality_summary(master: pd.DataFrame) -> None:
    incompletas = ((1 - master.isna().mean()) * 100).loc[lambda s: s < 100].sort_values()
    logger.info(f"=== {master.shape[1]} columnas; {len(incompletas)} con NaN (XGBoost los maneja nativo) ===")
    for col, pct in incompletas.items():
        logger.info(f"  {col:<40s} {pct:5.1f}%")

def main() -> None:
    dim = build_dim_municipios()
    save_parquet(dim, MASTER_DIR / "dim_municipios.parquet", partition_cols=None, compression=PARQUET_COMPRESSION)

    master = build_master_table(dim)
    out = MASTER_DIR / "master_table.parquet"
    save_parquet(master, out, partition_cols=None, compression=PARQUET_COMPRESSION)
    logger.info(f"master_table: {len(master)} filas, {master.shape[1]} columnas -> {out}")
    print_quality_summary(master)


if __name__ == "__main__":
    main()