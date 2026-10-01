from __future__ import annotations
import logging
import sys
from pathlib import Path
import pyarrow.dataset as ds
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry.base import BaseGeometry

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

try:
    from config.settings import BBOX_AGS
except ImportError:
    BBOX_AGS = None
    logger.warning("config.settings no tiene BBOX_AGS: se omite validación de bounding box de geometrías")

try:
    from config.settings import INEGI_MUNICIPIOS_AGS
except ImportError:
    INEGI_MUNICIPIOS_AGS = None
    logger.warning("config.settings no tiene INEGI_MUNICIPIOS_AGS: se omite validación de claves AGEM")

MUNICIPIOS_VALIDOS = set(MUNICIPIOS_AGS.keys())

RANGOS_NASA = {
    "T2M_MAX":            (-10, 50),
    "T2M_MIN":             (-15, 35),
    "T2M":                 (-15, 45),
    "RH2M":                (0, 100),
    "WS2M":                 (0, 40),      # m/s
    "ALLSKY_SFC_SW_DWN":    (0, 40),      # MJ/m2/día
    "PS":                  (65, 95),      # kPa, ajustado a ~1900 msnm
    "PRECTOTCORR":          (0, 300),     # mm/día
}
RANGO_CHIRPS = (0, 300)          # mm/día
RANGO_PM25 = (0, 1200)           # µg/m3
PM25_FLAG_SOSPECHOSO = 500       # marca, no borra

# Palabras a buscar en layer_name para aislar la(s) capa(s) de ciclones dentro
# del archivo combinado que produce cenapred.py (que junta TODAS las capas
# que matchean KEYWORDS_RIESGO en un solo parquet). Si ninguna capa descubierta
# contiene alguna de estas palabras, clean_cenapred lo reporta como warning.
KEYWORDS_CICLON = ["cicl"]


def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    before_na = df[col].isna().sum()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    nuevos_na = df[col].isna().sum() - before_na
    if nuevos_na > 0:
        logger.warning(f"{nuevos_na} fechas no parseables -> NaT (se dropean)")
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


def clean_municipios_geom(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "municipios_geom"
    df = df.copy()
    df = check_municipio_names(df, fuente=fuente)

    if "geometry_wkt" not in df.columns:
        logger.error(f"[{fuente}] No existe columna 'geometry_wkt', no se puede validar geometría")
        return df

    geoms: list[BaseGeometry | None] = []
    validas = []
    tipos_ok = []
    centroides_en_bbox = []

    for wkt_str in df["geometry_wkt"]:
        geom = None
        es_valida = False
        try:
            if pd.isna(wkt_str) or not str(wkt_str).strip():
                raise ValueError("WKT vacío")
            geom = shapely_wkt.loads(wkt_str)
            es_valida = geom.is_valid and not geom.is_empty
        except Exception as e:
            logger.error(f"[{fuente}] WKT no parseable: {e}")
        geoms.append(geom)
        validas.append(es_valida)

        tipo_ok = geom is not None and geom.geom_type in ("Polygon", "MultiPolygon")
        tipos_ok.append(tipo_ok)

        if geom is not None and es_valida and BBOX_AGS is not None:
            c = geom.centroid
            dentro = (
                BBOX_AGS["lon_min"] <= c.x <= BBOX_AGS["lon_max"]
                and BBOX_AGS["lat_min"] <= c.y <= BBOX_AGS["lat_max"]
            )
            centroides_en_bbox.append(dentro)
        else:
            centroides_en_bbox.append(None)

    df["geom_flag_valida"] = validas
    df["geom_flag_tipo_correcto"] = tipos_ok
    if BBOX_AGS is not None:
        df["geom_flag_centroide_en_bbox"] = centroides_en_bbox

    n_invalidas = (~pd.Series(validas)).sum()
    n_tipo_malo = (~pd.Series(tipos_ok)).sum()
    if n_invalidas > 0:
        logger.error(f"[{fuente}] {n_invalidas} geometrías inválidas o vacías (ver geom_flag_valida)")
    if n_tipo_malo > 0:
        logger.error(f"[{fuente}] {n_tipo_malo} geometrías que no son Polygon/MultiPolygon (ver geom_flag_tipo_correcto)")
    if BBOX_AGS is not None:
        n_fuera_bbox = sum(v is False for v in centroides_en_bbox)
        if n_fuera_bbox > 0:
            logger.error(
                f"[{fuente}] {n_fuera_bbox} centroides FUERA del bounding box de Aguascalientes "
                f"-> revisar CRS/coordenadas (ver geom_flag_centroide_en_bbox)"
            )

    if INEGI_MUNICIPIOS_AGS is not None and "cve_agem" in df.columns and "municipio" in df.columns:
        esperado = df["municipio"].map(INEGI_MUNICIPIOS_AGS)
        mismatch = esperado.notna() & (esperado != df["cve_agem"].astype(str))
        n_mismatch = mismatch.sum()
        if n_mismatch > 0:
            logger.error(
                f"[{fuente}] {n_mismatch} filas con cve_agem que NO coincide con el catálogo "
                f"INEGI_MUNICIPIOS_AGS (revisar: {df.loc[mismatch, ['municipio', 'cve_agem']].to_dict('records')})"
            )
        df["geom_flag_cve_agem_ok"] = ~mismatch

    df = drop_duplicates_report(df, subset=["municipio"], fuente=fuente)
    return df


def clean_cenapred(df: pd.DataFrame) -> pd.DataFrame:
    """
    cenapred.py descubre TODAS las capas cuyo nombre matchea KEYWORDS_RIESGO y las
    concatena en un solo parquet (con 'layer_id'/'layer_name' distinguiendo cada una).
    Esta limpieza está calibrada para una capa específica de "ciclones históricos"
    (columnas MPO_LEY/AREA/Ciclones_f/Ciclones_l) -- así que primero aislamos esa(s)
    capa(s) por nombre antes de aplicar esa lógica. Si ninguna capa descubierta por
    cenapred.py contiene 'cicl' en el nombre, probablemente KEYWORDS_RIESGO en
    cenapred.py no está encontrando la capa correcta y hay que ajustarlo ahí.
    """
    fuente = "cenapred_ciclones"
    df = df.copy()

    if "layer_name" in df.columns:
        capas_presentes = sorted(df["layer_name"].astype(str).unique().tolist())
        logger.info(f"[{fuente}] capas presentes en cenapred.parquet: {capas_presentes}")

        mask_ciclon = df["layer_name"].astype(str).str.contains(
            "|".join(KEYWORDS_CICLON), case=False, na=False
        )
        if mask_ciclon.any():
            df = df[mask_ciclon].copy()
            logger.info(f"[{fuente}] filtrado a {len(df)} filas de capa(s) de ciclones")
        else:
            logger.warning(
                f"[{fuente}] ninguna capa descubierta por cenapred.py contiene "
                f"{KEYWORDS_CICLON} en el nombre (capas disponibles arriba). "
                f"Los campos MPO_LEY/AREA/Ciclones_f probablemente NO existan en "
                f"este dataset -> revisa KEYWORDS_RIESGO en cenapred.py."
            )

    if "EDO_LEY" in df.columns:
        n_antes = len(df)
        df = df[df["EDO_LEY"].astype(str) == "Aguascalientes"].copy()
        n_fuera = n_antes - len(df)
        if n_fuera > 0:
            logger.warning(f"[{fuente}] {n_fuera} filas de otros estados eliminadas (EDO_LEY != Aguascalientes)")

    if "MPO_LEY" in df.columns:
        df = check_municipio_names(df, col="MPO_LEY", fuente=fuente)

    if "AREA" in df.columns:
        mask_area_mala = df["AREA"] <= 0
        n_area_mala = mask_area_mala.sum()
        df["cenapred_flag_area_invalida"] = mask_area_mala
        if n_area_mala > 0:
            logger.error(f"[{fuente}] {n_area_mala} filas con AREA <= 0 (dato corrupto, ver cenapred_flag_area_invalida)")

    if "Ciclones_f" in df.columns:
        valores_validos = {0, 1}
        vals = pd.to_numeric(df["Ciclones_f"], errors="coerce")
        mask_invalida = ~vals.isin(valores_validos)
        n_invalida = mask_invalida.sum()
        if n_invalida > 0:
            logger.warning(
                f"[{fuente}] {n_invalida} valores de 'Ciclones_f' fuera de {{0,1}} -> "
                f"se dejan en NaN en vez de asumir 0 (ver cenapred_flag_ciclones_valor_raro)"
            )
        df["cenapred_flag_ciclones_valor_raro"] = mask_invalida
        vals[mask_invalida] = np.nan
        df["Ciclones_f"] = vals

    subset_dedupe = [c for c in ("FID",) if c in df.columns] or [
        c for c in ("MPO_LEY", "layer_name") if c in df.columns
    ]
    if subset_dedupe:
        df = drop_duplicates_report(df, subset=subset_dedupe, fuente=fuente)

    return df

FUENTES_DINAMICAS = {
    "nasa_power": (clean_nasa_power, "nasa_power"),
    "chirps": (clean_chirps, "chirps"),
    "openaq": (clean_openaq, "openaq"),
}

FUENTES_ESTATICAS = [
    "poblacion",
    "elevacion",
    "rios",
    "geometria_municipios",
    "ciclones",
]

LIMPIADORES_ESTATICOS_ESPECIALES = {
    "geometria_municipios": clean_municipios_geom,
    "ciclones": clean_cenapred,
}

# FIX: estas rutas antes apuntaban a "inegi.parquet/<nombre>.parquet" y
# "cenapred.parquet/ciclones.parquet", que NUNCA existen -- inegi.py guarda cada
# tabla como su propio directorio particionado top-level (inegi_<nombre>.parquet),
# y cenapred.py guarda un único cenapred.parquet con todas las capas juntas.
MAPEO_RUTAS = {
    "poblacion": PROCESSED_DIR / "inegi_poblacion.parquet",
    "elevacion": PROCESSED_DIR / "inegi_elevacion.parquet",
    "rios": PROCESSED_DIR / "inegi_rios.parquet",
    "geometria_municipios": PROCESSED_DIR / "inegi_geometria_municipios.parquet",
    "ciclones": PROCESSED_DIR / "cenapred.parquet",
    "cenapred_ciclones": PROCESSED_DIR / "cenapred.parquet",
}


def _read_processed(nombre: str) -> pd.DataFrame | None:
    if nombre in MAPEO_RUTAS:
        path = MAPEO_RUTAS[nombre]
    else:
        path = PROCESSED_DIR / f"{nombre}.parquet"
    if not path.exists():
        logger.warning(f"No existe {path}, se omite")
        return None
    try:
        if path.is_dir():
            dataset = ds.dataset(path, format="parquet")
            return dataset.to_table().to_pandas()
        else:
            return pd.read_parquet(path, engine="pyarrow")
    except Exception as e:
        logger.error(f"Error al leer {path}: {e}")
        return None


def clean_dynamic_source(nombre: str, cleaner_fn, source_tag: str) -> pd.DataFrame | None:
    df = _read_processed(nombre)
    if df is None:
        return None

    logger.info(f"--- Limpiando {nombre} ({len(df)} filas antes, leído de data/processed/) ---")
    df_limpio = cleaner_fn(df)
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

    cleaner_especial = LIMPIADORES_ESTATICOS_ESPECIALES.get(nombre)
    if cleaner_especial is not None:
        df_limpio = cleaner_especial(df)
    else:
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