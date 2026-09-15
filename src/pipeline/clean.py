"""
ETAPA 2 — Limpieza de datos.

Se ejecuta DESPUÉS de la ingesta (nasa_power.py, chirps.py, openaq/sinaica.py, ...)
y ANTES de build_master_table.py.

REGLA DE ORO: data/processed/ es SOLO LECTURA en este script. Nunca se
sobreescribe. Los datos limpios se escriben en data/master/, con el mismo
esquema de particionado hive (source=X/year=Y) que ya usas en processed,
para que sigan siendo baratos de leer por año/fuente más adelante.

    data/processed/nasa_power.parquet/source=nasa_power/year=2020/...   (RAW, intocable)
    data/processed/chirps.parquet/source=chirps/year=2020/...           (RAW, intocable)
    data/processed/openaq.parquet/source=openaq/year=2020/...           (RAW, intocable)
                              │
                              ▼  clean_dynamic_source()
    data/master/nasa_power.parquet/source=nasa_power/year=2020/...      (LIMPIO)
    data/master/chirps.parquet/source=chirps/year=2020/...              (LIMPIO)
    data/master/openaq.parquet/source=openaq/year=2020/...              (LIMPIO)
    data/master/poblacion.parquet                                      (LIMPIO, estática, sin partición)
    data/master/elevacion.parquet
    data/master/distancia_rio.parquet
    data/master/municipios_geom.parquet
    data/master/cenapred_ciclones.parquet

build_master_table.py debe apuntar sus lecturas a data/master/, no a
data/processed/, una vez que este script haya corrido.

Qué limpia cada fuente dinámica (mismo criterio ya validado antes):

  NASA POWER  -> sentinelas -999, rangos físicos, T2M_MAX<T2M_MIN, dedupe
  CHIRPS      -> fechas string->datetime, precipitación negativa/extrema, dedupe
  OpenAQ/SINAICA (PM2.5) -> rangos físicos, flag de valores sospechosos >500, dedupe

Nada se elimina por estar fuera de rango: se pone en NaN + se deja una
columna `<col>_flag_outlier` (o `_flag_sospechoso` para PM2.5) como
evidencia, porque para Isolation Forest / XGBoost más adelante es más útil
saber "esto se corrigió" que perder la fila completa.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    PROCESSED_DIR,
    MASTER_DIR,
    MUNICIPIOS_AGS,
    FECHA_INICIO_MASTER,
    FECHA_FIN_MASTER,
    PARQUET_COMPRESSION,
)
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MUNICIPIOS_VALIDOS = set(MUNICIPIOS_AGS.keys())

RANGOS_NASA = {
    "T2M_MAX":            (-10, 50),
    "T2M_MIN":             (-15, 35),
    "T2M":                 (-15, 45),
    "RH2M":                (0, 100),
    "WS2M":                 (0, 40),      # m/s
    "ALLSKY_SFC_SW_DWN":    (0, 40),      # MJ/m2/día
    "PS":                  (65, 95),      # kPa, ajustado a 1900 msnm
    "PRECTOTCORR":          (0, 300),     # mm/día
}
RANGO_CHIRPS = (0, 300)          # mm/día
RANGO_PM25 = (0, 1200)           # µg/m3
PM25_FLAG_SOSPECHOSO = 500       # marca, no borra



def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    before_na = df[col].isna().sum()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    nuevos_na = df[col].isna().sum() - before_na
    if nuevos_na > 0:
        logger.warning(f"{nuevos_na} fechas no parseables NaT (se dropean)")
        df = df.dropna(subset=[col])
    return df


def check_municipio_names(df: pd.DataFrame, col: str = "municipio", fuente: str = "") -> pd.DataFrame:
    vistos = set(df[col].astype(str).unique())
    desconocidos = vistos - MUNICIPIOS_VALIDOS
    if desconocidos:
        logger.error(f"[{fuente}] Municipios NO reconocidos (revisa acentos/espacios): {desconocidos}")
    faltantes = MUNICIPIOS_VALIDOS - vistos
    if faltantes:
        logger.warning(f"[{fuente}] Municipios del catálogo ausentes en esta fuente: {faltantes}")
    return df


def drop_duplicates_report(df: pd.DataFrame, subset: list[str], fuente: str = "") -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep="last")
    n_removed = n_before - len(df)
    if n_removed > 0:
        logger.warning(f"[{fuente}] {n_removed} filas duplicadas en {subset} eliminadas (keep='last')")
    return df


def replace_sentinels(df: pd.DataFrame, cols: list[str], threshold: float = -900, fuente: str = "") -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        mask = df[col] <= threshold
        n = mask.sum()
        if n > 0:
            logger.warning(f"[{fuente}] {n} sentinelas (<= {threshold}) en '{col}' -> NaN")
            df.loc[mask, col] = np.nan
    return df


def flag_out_of_range(df: pd.DataFrame, col: str, lo: float, hi: float, fuente: str = "") -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    mask = (df[col] < lo) | (df[col] > hi)
    n = mask.sum()
    flag_col = f"{col}_flag_outlier"
    df[flag_col] = mask
    if n > 0:
        logger.warning(f"[{fuente}] {n} valores de '{col}' fuera de [{lo},{hi}] -> NaN + flag en {flag_col}")
        df.loc[mask, col] = np.nan
    return df


def coverage_report(
    df: pd.DataFrame,
    fuente: str,
    date_col: str = "date",
    municipio_col: str = "municipio",
    start=FECHA_INICIO_MASTER,
    end=FECHA_FIN_MASTER,
) -> None:
    full_days = pd.date_range(start, end, freq="D")
    expected = len(full_days) * len(MUNICIPIOS_VALIDOS)
    en_rango = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))]
    present_en_rango = en_rango.drop_duplicates(subset=[date_col, municipio_col]).shape[0]

    logger.info(
        f"[{fuente}] cobertura: {present_en_rango}/{expected} celdas "
        f"({present_en_rango / expected * 100:.1f}%) dentro de {start}..{end} "
        f"| filas totales en dataset: {len(df)}"
    )
    if present_en_rango < expected:
        cobertura_por_municipio = (
            en_rango.groupby(municipio_col, observed=True)[date_col].nunique().sort_values()
        )
        logger.info(f"[{fuente}] municipios con menos días cubiertos: {dict(cobertura_por_municipio.head(3))}")


# ──────────────────────────────────────────────────────────────
# Limpieza específica por fuente
# ──────────────────────────────────────────────────────────────

def clean_nasa_power(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "nasa_power"
    df = parse_dates(df, "date")
    df = check_municipio_names(df, fuente=fuente)
    df = replace_sentinels(df, list(RANGOS_NASA.keys()), fuente=fuente)

    for col, (lo, hi) in RANGOS_NASA.items():
        df = flag_out_of_range(df, col, lo, hi, fuente=fuente)

    if {"T2M_MAX", "T2M_MIN"}.issubset(df.columns):
        mask = df["T2M_MAX"] < df["T2M_MIN"]
        n = mask.sum()
        if n > 0:
            logger.warning(f"[{fuente}] {n} filas con T2M_MAX < T2M_MIN -> ambas a NaN")
            df.loc[mask, ["T2M_MAX", "T2M_MIN"]] = np.nan

    df = drop_duplicates_report(df, subset=["date", "municipio"], fuente=fuente)
    coverage_report(df, fuente)
    return df


def clean_chirps(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "chirps"
    df = parse_dates(df, "date")
    df = check_municipio_names(df, fuente=fuente)
    df = flag_out_of_range(df, "precipitation_mm", *RANGO_CHIRPS, fuente=fuente)
    df = drop_duplicates_report(df, subset=["date", "municipio"], fuente=fuente)
    coverage_report(df, fuente)
    return df


def clean_openaq(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza de calidad del aire (carpeta openaq.parquet, dato real = SINAICA)."""
    fuente = "openaq"
    df = parse_dates(df, "date")
    df = check_municipio_names(df, fuente=fuente)

    for col in ("pm25_mean", "pm25_max", "pm25_p95"):
        if col not in df.columns:
            continue
        df = flag_out_of_range(df, col, *RANGO_PM25, fuente=fuente)
        flag_col = f"{col}_flag_sospechoso"
        df[flag_col] = df[col] > PM25_FLAG_SOSPECHOSO
        n_sosp = df[flag_col].sum()
        if n_sosp > 0:
            logger.warning(
                f"[{fuente}] {n_sosp} valores de '{col}' > {PM25_FLAG_SOSPECHOSO} µg/m3 "
                f"marcados en {flag_col} (NO se eliminan, revisar a mano)"
            )

    df = drop_duplicates_report(df, subset=["date", "municipio"], fuente=fuente)
    coverage_report(df, fuente)

    sin_estacion = MUNICIPIOS_VALIDOS - set(df["municipio"].astype(str).unique())
    if sin_estacion:
        logger.warning(f"[{fuente}] Sin estación, quedarán NaN en la tabla maestra: {sin_estacion}")
    return df


def clean_static_table(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """
    Población, elevación, distancia a río, geometrías, CENAPRED: solo valida
    catálogo. No todas usan la misma columna de municipio (ej. CENAPRED trae
    'MPO_LEY' en vez de 'municipio'), así que se detecta cuál usar en vez de
    asumirla y tronar.
    """
    col_municipio = "municipio" if "municipio" in df.columns else None
    if col_municipio is None:
        for candidata in ("MPO_LEY", "NOM_MUN", "nom_agem"):
            if candidata in df.columns:
                col_municipio = candidata
                break

    if col_municipio is None:
        logger.warning(
            f"[{fuente}] No se encontró columna de municipio reconocible "
            f"(columnas disponibles: {list(df.columns)}). Se omite validación de catálogo."
        )
        return df

    df = check_municipio_names(df, col=col_municipio, fuente=fuente)
    n_dup = df.duplicated(subset=[col_municipio]).sum()
    if n_dup > 0:
        logger.warning(f"[{fuente}] {n_dup} municipios duplicados en '{col_municipio}' (revisar a mano)")
    return df


# ──────────────────────────────────────────────────────────────
# Orquestador
# ──────────────────────────────────────────────────────────────

# nombre de carpeta/archivo -> (función de limpieza, tag de 'source' para particionar)
FUENTES_DINAMICAS = {
    "nasa_power": (clean_nasa_power, "nasa_power"),
    "chirps": (clean_chirps, "chirps"),
    "openaq": (clean_openaq, "openaq"),
}

# tablas estáticas: sin fecha, sin partición, un archivo plano en data/master
FUENTES_ESTATICAS = [
    "poblacion",
    "elevacion",
    "distancia_rio",
    "municipios_geom",       # ajusta el nombre si tu archivo INEGI se llama distinto
    "cenapred_ciclones",     # ajusta el nombre si tu archivo CENAPRED se llama distinto
]


def _read_processed(nombre: str) -> pd.DataFrame | None:
    """
    Lee data/processed/<nombre>.parquet, sea un archivo plano o un dataset
    particionado hive (source=X/year=Y/*.parquet). pyarrow detecta ambos
    casos automáticamente al recibir la ruta del directorio/archivo.
    """
    path = PROCESSED_DIR / f"{nombre}.parquet"
    if not path.exists():
        logger.warning(f"No existe {path}, se omite")
        return None
    return pd.read_parquet(path, engine="pyarrow")


def clean_dynamic_source(nombre: str, cleaner_fn, source_tag: str) -> pd.DataFrame | None:
    df = _read_processed(nombre)
    if df is None:
        return None

    logger.info(f"--- Limpiando {nombre} ({len(df)} filas antes, leído de data/processed/) ---")
    df_limpio = cleaner_fn(df)

    # recalcular year/source explícitos para particionar de forma consistente,
    # sin depender de que ya vinieran bien como columnas de partición al leer
    df_limpio = df_limpio.copy()
    df_limpio["year"] = df_limpio["date"].dt.year
    df_limpio["source"] = source_tag

    out_path = MASTER_DIR / f"{nombre}.parquet"
    save_parquet(
        df_limpio,
        out_path,
        partition_cols=["source", "year"],
        compression=PARQUET_COMPRESSION,
    )
    logger.info(f"--- {nombre}: {len(df_limpio)} filas después -> {out_path} (particionado) ---")
    return df_limpio


def clean_static_source(nombre: str) -> pd.DataFrame | None:
    df = _read_processed(nombre)
    if df is None:
        return None

    logger.info(f"--- Limpiando {nombre} ({len(df)} filas, tabla estática) ---")
    df_limpio = clean_static_table(df, nombre)

    out_path = MASTER_DIR / f"{nombre}.parquet"
    save_parquet(df_limpio, out_path, partition_cols=None, compression=PARQUET_COMPRESSION)
    logger.info(f"--- {nombre}: guardado sin particionar -> {out_path} ---")
    return df_limpio


def run_cleaning_pipeline() -> None:
    for nombre, (fn, tag) in FUENTES_DINAMICAS.items():
        clean_dynamic_source(nombre, fn, tag)

    for nombre in FUENTES_ESTATICAS:
        clean_static_source(nombre)

    logger.info(
        "Limpieza completa. data/processed/ no fue modificado. "
        "Revisa los WARNING/ERROR arriba, luego apunta build_master_table.py a data/master/."
    )


if __name__ == "__main__":
    run_cleaning_pipeline()