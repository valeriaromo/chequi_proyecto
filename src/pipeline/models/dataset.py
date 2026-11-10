from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import model_config as cfg

# Cargar el dataset de entrenamiento para un objetivo y horizonte, con features y target


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    year: pd.Series
    persistence: Optional[pd.Series]  # T2M_MIN regression baseline de hoy


def feature_group(col: str) -> str:
    if "aod" in col.lower():
        return "aod"
    if col in cfg.STATIC_COLUMNS:
        return "static"
    if col in cfg.CYCLONE_COLUMNS:
        return "cyclone"
    return "base"


def _with_static(df, feature_set):
    extra = [c for c in cfg.EXTRA_BY_SET[feature_set] if c not in df.columns]
    if not extra:
        return df
    dim = pd.read_parquet(cfg.MASTER_DIR / "dim_municipios.parquet")[["municipio", *extra]]
    return df.merge(dim, on="municipio", how="left", validate="many_to_one")


def _feature_columns(df, feature_set):
    use_aod = feature_set == "aod"  # solo el set "aod" incluye AOD; "all" = todo menos AOD
    cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in cfg.KEY_COLS
        and "target" not in c.lower()
        and df[c].nunique(dropna=True) > 1
        and (use_aod or "aod" not in c.lower())
    ]
    if not use_aod and any("aod" in c.lower() for c in cols):
        raise AssertionError("AOD column leaked into features")
    return cols


def build_dataset(task: str, table: str, feature_set: str) -> Dataset:
    # Cargar tabla y ordenar por municipio y fecha
    df = (
        pd.read_parquet(cfg.TABLES[table])
        .sort_values(["municipio", "date"])
        .reset_index(drop=True)
    )

    # Incorporar variables estáticas según el feature_set
    df = _with_static(df, feature_set)

    source = cfg.RAIN_COL if task == "classification" else cfg.TEMP_COL
    if source not in df.columns:
        raise KeyError(
            f"Column '{source}' not found; "
            f"set it in model_config.py. Available: {list(df.columns)}"
        )

    # Obtener el target futuro mediante shift(-HORIZON)
    future = df.groupby("municipio")[source].shift(-cfg.HORIZON)
    keep = future.notna()
    df = df[keep].reset_index(drop=True)
    future = future[keep].reset_index(drop=True)

    is_clf = task == "classification"

    return Dataset(
        X=df[_feature_columns(df, feature_set)],
        y=(future >= cfg.RAIN_THRESHOLD_MM).astype(int) if is_clf else future,
        year=pd.to_datetime(df["date"]).dt.year,
        persistence=None if is_clf else df[cfg.TEMP_COL],
    )