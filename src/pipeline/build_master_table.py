from __future__ import annotations
import logging
import sys
from pathlib import Path
from clean import FUENTES_DINAMICAS, FUENTES_ESTATICAS

FUENTES_ESTATICAS_ESCALARES = [
    f for f in FUENTES_ESTATICAS if f not in ("geometria_municipios", "ciclones")
]
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    MASTER_DIR,
    MUNICIPIOS_AGS,
    FECHA_INICIO_MASTER,
    FECHA_FIN_MASTER,
    PARQUET_COMPRESSION,
)
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MUNICIPIOS_VALIDOS = list(MUNICIPIOS_AGS.keys())

RAIN_WINDOWS = [(1, "1d"), (3, "3d"), (7, "7d"), (30, "30d"), (90, "90d")]


def _read_master(nombre: str) -> pd.DataFrame | None:
    path = MASTER_DIR / f"{nombre}.parquet"
    if not path.exists():
        logger.warning(f"No existe {path}, se omite esa fuente")
        return None
        
    df = pd.read_parquet(path, engine="pyarrow")
    df = df.drop(columns=[c for c in ("source", "year") if c in df.columns])
    if "date" in df.columns and "municipio" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["municipio", "date"], keep="last")
        if len(df) < n_before:
            logger.warning(f"[{nombre}] Se eliminaron {n_before - len(df)} filas duplicadas al leer de master")
    elif "municipio" in df.columns:
        n_before = len(df)
        df = df.drop_duplicates(subset=["municipio"], keep="last")
        if len(df) < n_before:
            logger.warning(f"[{nombre}] Se eliminaron {n_before - len(df)} municipios duplicados al leer de master")
            
    return df


def build_canonical_grid() -> pd.DataFrame:
    dias = pd.date_range(FECHA_INICIO_MASTER, FECHA_FIN_MASTER, freq="D")
    grid = pd.MultiIndex.from_product(
        [MUNICIPIOS_VALIDOS, dias], names=["municipio", "date"]
    ).to_frame(index=False)
    logger.info(f"Grid canónico: {len(grid)} filas ({len(MUNICIPIOS_VALIDOS)} municipios x {len(dias)} días)")
    return grid


def add_rain_accumulations(master: pd.DataFrame) -> pd.DataFrame:
    if "precipitation_mm" not in master.columns:
        return master

    master = master.sort_values(["municipio", "date"])
    grouped = master.groupby("municipio")["precipitation_mm"]

    for window, label in RAIN_WINDOWS:
        master[f"rain_{label}"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).sum()
        )
        master[f"rain_{label}_valid_days"] = grouped.transform(
            lambda s, w=window: s.rolling(w, min_periods=1).count()
        )

    return master

def build_master_table() -> pd.DataFrame:
    master = build_canonical_grid()

    for nombre in FUENTES_DINAMICAS:
        df = _read_master(nombre)
        if df is None:
            continue
        n_before = len(master)
        master = master.merge(df, on=["date", "municipio"], how="left")
        assert len(master) == n_before, f"El merge de {nombre} cambió el número de filas del grid (hay duplicados en la fuente)"
        logger.info(f"Integrado {nombre}: {df.shape[1] - 2} columnas nuevas")

    master = add_rain_accumulations(master)

    for nombre in FUENTES_ESTATICAS_ESCALARES:
        df = _read_master(nombre)
        if df is None:
            continue
        n_before = len(master)
        master = master.merge(df, on="municipio", how="left")
        assert len(master) == n_before, f"El merge de {nombre} cambió el número de filas (hay municipios duplicados en la fuente)"
        logger.info(f"Integrado {nombre} (estático): {df.shape[1] - 1} columnas nuevas, broadcast a todas las fechas")

    cenapred = _read_master("ciclones")
    if cenapred is not None and "MPO_LEY" in cenapred.columns:
        cenapred_flag = (
            cenapred.rename(columns={"MPO_LEY": "municipio"})[["municipio", "Ciclones_f"]]
            .rename(columns={"Ciclones_f": "ciclon_declarado_historico"})
            .drop_duplicates(subset=["municipio"])
        )
        n_before = len(master)
        master = master.merge(cenapred_flag, on="municipio", how="left")
        assert len(master) == n_before, "El merge de cenapred_ciclones cambió el número de filas"
        logger.info("Integrado cenapred_ciclones: 1 columna nueva (ciclon_declarado_historico)")
    elif cenapred is not None:
        logger.warning(
            f"cenapred_ciclones no tiene columna 'MPO_LEY' (columnas: {list(cenapred.columns)}); "
            f"se omite su integración, ajusta el nombre de columna en este script"
        )

    master = master.sort_values(["municipio", "date"]).reset_index(drop=True)
    master["year"] = master["date"].dt.year  # útil para walk-forward validation (Etapa 29)

    return master


def build_dim_municipios() -> pd.DataFrame | None:
    geom = _read_master("geometria_municipios")
    if geom is None:
        return None

    dim = geom.copy()
    for nombre in FUENTES_ESTATICAS_ESCALARES:
        df = _read_master(nombre)
        if df is None:
            continue
        dim = dim.merge(df, on="municipio", how="left")

    cenapred = _read_master("ciclones")
    if cenapred is not None and "MPO_LEY" in cenapred.columns:
        cenapred_flag = (
            cenapred.rename(columns={"MPO_LEY": "municipio"})[["municipio", "Ciclones_f", "Ciclones_l"]]
            .rename(columns={"Ciclones_f": "ciclon_declarado_historico", "Ciclones_l": "ciclon_descripcion"})
            .drop_duplicates(subset=["municipio"])
        )
        dim = dim.merge(cenapred_flag, on="municipio", how="left")

    logger.info(f"dim_municipios: {len(dim)} filas, {dim.shape[1]} columnas")
    return dim


def print_quality_summary(master: pd.DataFrame) -> None:
    logger.info("=== Resumen de completitud del master_table (% de filas con dato) ===")
    completitud = (1 - master.isna().mean()) * 100
    for col, pct in completitud.sort_values().items():
        if col in ("municipio", "date", "year"):
            continue
        logger.info(f"  {col:<35s} {pct:5.1f}%")


def main() -> None:
    master = build_master_table()
    out_path = MASTER_DIR / "master_table.parquet"
    save_parquet(master, out_path, partition_cols=None, compression=PARQUET_COMPRESSION)
    logger.info(f"master_table: {len(master)} filas, {master.shape[1]} columnas -> {out_path}")
    print_quality_summary(master)

    dim = build_dim_municipios()
    if dim is not None:
        dim_path = MASTER_DIR / "dim_municipios.parquet"
        save_parquet(dim, dim_path, partition_cols=None, compression=PARQUET_COMPRESSION)
        logger.info(f"dim_municipios: {len(dim)} filas, {dim.shape[1]} columnas -> {dim_path}")


if __name__ == "__main__":
    main()