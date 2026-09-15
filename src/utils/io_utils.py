"""
Utilidades de entrada/salida.

La idea central de este módulo: cada vez que se guarda un DataFrame,
se reduce su tamaño en memoria/disco ANTES de escribirlo (downcasting
de tipos + compresión zstd + particionado). Esto es lo que evita que
el proyecto termine con una "db pesada".
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el tamaño en memoria de un DataFrame:
    - float64 -> float32
    - int64   -> int32/int16 según el rango
    - columnas de baja cardinalidad (ej. municipio) -> category

    Esto solo puede reducir 40-60% el tamaño sin perder precisión relevante
    para variables climáticas (no necesitamos float64 para temperatura).
    """
    before_mb = df.memory_usage(deep=True).sum() / 1e6

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["object"]).columns:
        n_unique = df[col].nunique()
        if n_unique / max(len(df), 1) < 0.5:  # baja cardinalidad -> category
            df[col] = df[col].astype("category")

    after_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(f"Downcast: {before_mb:.2f} MB -> {after_mb:.2f} MB "
                f"({(1 - after_mb / before_mb) * 100:.1f}% reducción)" if before_mb else "")
    return df


def save_parquet(
    df: pd.DataFrame,
    path: Path,
    partition_cols: list[str] | None = None,
    compression: str = "zstd",
) -> None:
    """
    Guarda un DataFrame en Parquet comprimido, con downcasting previo.
    Si se pasan partition_cols, se particiona (ej. por año) para que
    lecturas futuras no tengan que cargar el histórico completo.
    """
    df = downcast_dtypes(df.copy())
    path.parent.mkdir(parents=True, exist_ok=True)

    if partition_cols:
        df.to_parquet(
            path,
            engine="pyarrow",
            compression=compression,
            partition_cols=partition_cols,
            index=False,
        )
    else:
        df.to_parquet(
            path,
            engine="pyarrow",
            compression=compression,
            index=False,
        )

    size_mb = _dir_or_file_size_mb(path)
    logger.info(f"Guardado: {path} ({size_mb:.3f} MB en disco)")


def append_parquet(df: pd.DataFrame, path: Path, compression: str = "zstd") -> None:
    """
    Agrega filas nuevas a un parquet existente sin reescribir todo el histórico.
    Útil para actualizaciones incrementales (ej. correr la ingesta cada día).
    """
    df = downcast_dtypes(df.copy())
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=[c for c in ("date", "municipio") if c in combined.columns],
            keep="last",
        )
    else:
        combined = df

    save_parquet(combined, path, compression=compression)


def _dir_or_file_size_mb(path: Path) -> float:
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    elif path.exists():
        total = path.stat().st_size
    else:
        total = 0
    return total / 1e6


def cleanup_raw(path: Path) -> None:
    """
    Borra archivos crudos temporales (ej. rasters descargados) una vez que
    ya se extrajo y guardó el resultado agregado. Esto es lo que evita
    acumular TB de rasters que solo se usan una vez.
    """
    if path.exists():
        path.unlink()
        logger.info(f"Crudo eliminado: {path}")
