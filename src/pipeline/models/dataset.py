from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import model_config as cfg

# cargar el dataset de entrenamiento para un objetivo y horizonte, con features y target

@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    year: pd.Series
    persistence: Optional[pd.Series]  # T2M_MIN regression baseline de hoy


def _feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include="number").columns
    return [
        c for c in numeric
        if c not in cfg.KEY_COLS and not any(s in c.lower() for s in cfg.EXCLUDED_SUBSTRINGS)
    ]


def build_dataset(task: str, table: str) -> Dataset:
    # se agrupa por municipio y se ordena por fecha para poder hacer shift(-HORIZON) y obtener el target futuro
    df = pd.read_parquet(cfg.TABLES[table]).sort_values(["municipio", "date"]).reset_index(drop=True)
    source = cfg.RAIN_COL if task == "classification" else cfg.TEMP_COL
    if source not in df.columns:
        raise KeyError(f"Column '{source}' not found; set it in model_config.py. Available: {list(df.columns)}")

    future = df.groupby("municipio")[source].shift(-cfg.HORIZON)
    keep = future.notna()
    df, future = df[keep].reset_index(drop=True), future[keep].reset_index(drop=True)

    is_clf = task == "classification"
    return Dataset(
        X=df[_feature_columns(df)],
        y=(future >= cfg.RAIN_THRESHOLD_MM).astype(int) if is_clf else future,
        year=pd.to_datetime(df["date"]).dt.year,
        persistence=None if is_clf else df[cfg.TEMP_COL],
    )
