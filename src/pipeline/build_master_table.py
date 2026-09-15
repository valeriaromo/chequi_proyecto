"""
Construye la tabla maestra: municipio x día, con columnas de las 3 fuentes.

Este script asume que nasa_power.py, chirps.py y openaq.py ya corrieron
y dejaron sus resultados en data/processed/*.parquet.

Lee SOLO los parquet ya agregados (pequeños), nunca datos crudos.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import PROCESSED_DIR, MASTER_DIR, PARQUET_COMPRESSION
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _read_source(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    if not path.exists():
        logger.warning(f"No existe {path}, se omite esa fuente")
        return pd.DataFrame(columns=["date", "municipio"])
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.drop(columns=[c for c in ("source", "year") if c in df.columns])


def build_master_table() -> pd.DataFrame:
    nasa = _read_source("nasa_power")
    chirps = _read_source("chirps")
    openaq = _read_source("openaq")

    master = nasa.merge(chirps, on=["date", "municipio"], how="outer")
    master = master.merge(openaq, on=["date", "municipio"], how="outer")

    master = master.sort_values(["municipio", "date"]).reset_index(drop=True)

    # Features derivados de precipitación acumulada (fundamentales para flood/drought risk)
    if "precipitation_mm" in master.columns:
        master = master.sort_values(["municipio", "date"])
        for window, label in [(1, "1d"), (3, "3d"), (7, "7d"), (30, "30d"), (90, "90d")]:
            master[f"rain_{label}"] = (
                master.groupby("municipio")["precipitation_mm"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
            )

    out_path = MASTER_DIR / "master_table.parquet"
    save_parquet(master, out_path, partition_cols=None, compression=PARQUET_COMPRESSION)
    logger.info(f"Tabla maestra: {len(master)} filas, {master.shape[1]} columnas -> {out_path}")
    return master


if __name__ == "__main__":
    build_master_table()
