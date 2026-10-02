from __future__ import annotations
import logging
import re
import shutil
import sys
import unicodedata
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from shapely import wkt as shapely_wkt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (FECHA_FIN_MASTER,FECHA_INICIO_MASTER,MASTER_DIR,MUNICIPIOS_AGS,NASA_POWER_PARAMS,PARQUET_COMPRESSION,PROCESSED_DIR,)
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

# para la implementación de XGBoost, se requiere que el master tenga una fila por municipio
RANGOS_NASA = {
    "T2M_MAX": (-10, 50),
    "T2M_MIN": (-15, 35),
    "T2M": (-15, 45),
    "RH2M": (0, 100),
    "WS2M": (0, 40),
    "ALLSKY_SFC_SW_DWN": (0, 40),
    "PS": (65, 95),            # kPa 1900 msnm
    "PRECTOTCORR": (0, 300),   # mm/día
}
RANGO_CHIRPS = (0, 300)        # mm/día
RANGO_PM25 = (0, 1000)         # µg/m3 (coincide con el filtro de openaq.py)
PM25_FLAG_SOSPECHOSO = 500     # marca no borra
COLS_PM25 = ("pm25_mean", "pm25_max", "pm25_q98")
RANGO_AOD = (0, 5)             # AOD 550 nm (mismo tope que maiac.py)
RANGO_PCT = (0, 100)
AOD_MIN_PIXELES = 10           # menos píxeles válidos se flaggea
COLS_AOD = ("aod_055_mean", "aod_055_count", "aod_qa_best_pct")
KEYWORDS_CICLON = ("cicl",)
SENTINELA_UMBRAL = -900


def _normalizar(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()
CATALOGO_NORM = {_normalizar(m): m for m in MUNICIPIOS_AGS}

def _borrar(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()

def require_columns(df: pd.DataFrame, cols: list[str], fuente: str) -> None:
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        raise ValueError(f"[{fuente}] faltan columnas {faltan}. Disponibles: {list(df.columns)}")

def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    n_nat = int(df[col].isna().sum())
    if n_nat > 0:
        logger.warning(f"{n_nat} fechas no parseables/nulas -> filas descartadas")
        df = df.dropna(subset=[col])
    return df

def normalize_municipio(df: pd.DataFrame, col: str = "municipio", fuente: str = "") -> pd.DataFrame:
    df = df.copy()
    original = df[col].astype(str)
    mapeado = original.map(lambda v: CATALOGO_NORM.get(_normalizar(v)))
    corregidos = int((mapeado.notna() & (mapeado != original)).sum())
    if corregidos:
        logger.warning(f"[{fuente}] {corregidos} nombres de municipio normalizados al catálogo")
    desconocidos = set(original[mapeado.isna()].unique())
    if desconocidos:
        logger.error(
            f"[{fuente}] Municipios NO reconocidos, {int(mapeado.isna().sum())} filas descartadas: {desconocidos}"
        )

    df = df.loc[mapeado.notna()].copy()
    df["municipio"] = mapeado[mapeado.notna()].astype(str)
    faltantes = MUNICIPIOS_VALIDOS - set(df["municipio"].unique())
    if faltantes:
        logger.warning(f"[{fuente}] Municipios del catálogo ausentes: {sorted(faltantes)}")
    return df

def drop_duplicates_report(df: pd.DataFrame, subset: list[str], fuente: str = "") -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep="last")
    if n_before - len(df) > 0:
        logger.warning(f"[{fuente}] {n_before - len(df)} filas duplicadas en {subset} eliminadas (keep='last')")
    return df

def replace_sentinels(df: pd.DataFrame, cols: list[str], threshold: float = SENTINELA_UMBRAL, fuente: str = "") -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = df[col] <= threshold
        if mask.sum() > 0:
            logger.warning(f"[{fuente}] {int(mask.sum())} sentinelas (<= {threshold}) en '{col}' -> NaN")
            df.loc[mask, col] = np.nan
    return df

def flag_out_of_range(df: pd.DataFrame, col: str, lo: float, hi: float, fuente: str = "") -> pd.DataFrame:
    df = df.copy() # el Nan ayuda al XGBoost a no aprender de valores fuera de rango
    df[col] = pd.to_numeric(df[col], errors="coerce")
    mask = (df[col] < lo) | (df[col] > hi)
    df[f"qc_{col}_outlier"] = mask
    if mask.sum() > 0:
        logger.warning(f"[{fuente}] {int(mask.sum())} valores de '{col}' fuera de [{lo},{hi}] -> NaN (qc_{col}_outlier)")
        df.loc[mask, col] = np.nan
    return df

def warn_all_nan(df: pd.DataFrame, cols: list[str], fuente: str) -> None:
    for c in cols:
        if c in df.columns and df[c].isna().all():
            logger.error(f"[{fuente}] la columna '{c}' está 100% vacía tras la limpieza")

def coverage_report(df: pd.DataFrame, fuente: str, date_col: str = "date", municipio_col: str = "municipio",
                    start=FECHA_INICIO_MASTER, end=FECHA_FIN_MASTER) -> None:
    expected = len(pd.date_range(start, end, freq="D")) * len(MUNICIPIOS_VALIDOS)
    en_rango = df[(df[date_col] >= pd.Timestamp(start)) & (df[date_col] <= pd.Timestamp(end))]
    presentes = en_rango.drop_duplicates(subset=[date_col, municipio_col]).shape[0]
    logger.info(
        f"[{fuente}] cobertura: {presentes}/{expected} celdas ({presentes / expected * 100:.1f}%) "
        f"dentro de {start}..{end} | filas totales: {len(df)}"
    )
    if presentes < expected:
        por_mun = en_rango.groupby(municipio_col)[date_col].nunique().sort_values()
        logger.info(f"[{fuente}] municipios con menos días cubiertos: {dict(por_mun.head(3))}")


def clean_nasa_power(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "nasa_power"
    cols = list(NASA_POWER_PARAMS)
    require_columns(df, ["date", "municipio"] + cols, fuente)

    sin_rango = [c for c in cols if c not in RANGOS_NASA]
    if sin_rango:
        logger.warning(f"[{fuente}] variables sin rango físico definido (no validadas): {sin_rango}")

    df = parse_dates(df, "date")
    df = normalize_municipio(df, fuente=fuente)
    df = replace_sentinels(df, cols, fuente=fuente)
    for col in cols:
        if col in RANGOS_NASA:
            df = flag_out_of_range(df, col, *RANGOS_NASA[col], fuente=fuente)

    if {"T2M_MAX", "T2M_MIN"}.issubset(df.columns):
        mask = df["T2M_MAX"] < df["T2M_MIN"]
        df["qc_t2m_inconsistente"] = mask
        if mask.sum() > 0:
            logger.warning(f"[{fuente}] {int(mask.sum())} filas con T2M_MAX < T2M_MIN -> ambas a NaN")
            df.loc[mask, ["T2M_MAX", "T2M_MIN"]] = np.nan

    df = drop_duplicates_report(df, ["date", "municipio"], fuente)
    warn_all_nan(df, cols, fuente)
    coverage_report(df, fuente)
    return df


def clean_chirps(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "chirps"
    require_columns(df, ["date", "municipio", "precipitation_mm"], fuente)
    df = parse_dates(df, "date")
    df = normalize_municipio(df, fuente=fuente)
    df = replace_sentinels(df, ["precipitation_mm"], fuente=fuente)  # CHIRPS usa -9999 como nodata
    df = flag_out_of_range(df, "precipitation_mm", *RANGO_CHIRPS, fuente=fuente)
    df = drop_duplicates_report(df, ["date", "municipio"], fuente)
    warn_all_nan(df, ["precipitation_mm"], fuente)
    coverage_report(df, fuente)
    return df


def clean_openaq(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "openaq"
    require_columns(df, ["date", "municipio", *COLS_PM25], fuente)  # q98 viene de openaq.py (antes pm25_p95)
    df = parse_dates(df, "date")
    df = normalize_municipio(df, fuente=fuente)
    df = replace_sentinels(df, list(COLS_PM25), fuente=fuente)

    for col in COLS_PM25:
        df = flag_out_of_range(df, col, *RANGO_PM25, fuente=fuente)
        sosp = df[col] > PM25_FLAG_SOSPECHOSO
        df[f"qc_{col}_sospechoso"] = sosp
        if sosp.sum() > 0:
            logger.warning(f"[{fuente}] {int(sosp.sum())} valores de '{col}' > {PM25_FLAG_SOSPECHOSO} (marcados, no eliminados)")

    df = drop_duplicates_report(df, ["date", "municipio"], fuente)
    warn_all_nan(df, list(COLS_PM25), fuente)
    coverage_report(df, fuente)
    return df

def clean_maiac_aod(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "maiac_aod"
    require_columns(df, ["date", "municipio", *COLS_AOD], fuente)
    df = parse_dates(df, "date")
    df = normalize_municipio(df, fuente=fuente)
    df = replace_sentinels(df, list(COLS_AOD), fuente=fuente)
    neg = df["aod_055_count"] < 0
    if neg.any():
        logger.error(f"[{fuente}] {int(neg.sum())} conteos negativos en 'aod_055_count' -> NaN")
        df.loc[neg, "aod_055_count"] = np.nan
    sin_pixeles = df["aod_055_count"] == 0
    con_valor = df["aod_055_mean"].notna() | df["aod_qa_best_pct"].notna()
    incoherentes = sin_pixeles & con_valor
    if incoherentes.any():
        logger.error(f"[{fuente}] {int(incoherentes.sum())} filas con count=0 pero con AOD/best_pct -> NaN")
        df.loc[incoherentes, ["aod_055_mean", "aod_qa_best_pct"]] = np.nan

    df = flag_out_of_range(df, "aod_055_mean", *RANGO_AOD, fuente=fuente)
    df = flag_out_of_range(df, "aod_qa_best_pct", *RANGO_PCT, fuente=fuente)

    pocos = (df["aod_055_count"] > 0) & (df["aod_055_count"] < AOD_MIN_PIXELES)
    df["qc_aod_pocos_pixeles"] = pocos
    if pocos.any():
        logger.warning(f"[{fuente}] {int(pocos.sum())} celdas con < {AOD_MIN_PIXELES} píxeles válidos (marcadas, no eliminadas)")
    if "qc_aod_imputed" not in df.columns:
        df["qc_aod_imputed"] = False
    if "aod_imputed_method" not in df.columns:
        df["aod_imputed_method"] = "none"
    df["qc_aod_imputed"] = df["qc_aod_imputed"].fillna(False).astype(bool)
    df["aod_imputed_method"] = df["aod_imputed_method"].fillna("none").astype(str)
    inconsistente = df["qc_aod_imputed"] != (df["aod_imputed_method"] != "none")
    if inconsistente.any():
        logger.error(f"[{fuente}] {int(inconsistente.sum())} filas con qc_aod_imputed inconsistente con aod_imputed_method")

    df = drop_duplicates_report(df, ["date", "municipio"], fuente)
    valido = df["aod_055_mean"].notna()
    logger.info(f"[{fuente}] recuperación AOD: {valido.mean() * 100:.1f}% de celdas con AOD válido")
    por_mun = (valido.groupby(df["municipio"]).mean() * 100).sort_values().round(1)
    logger.info(f"[{fuente}] menor recuperación por municipio: {dict(por_mun.head(3))}")

    warn_all_nan(df, ["aod_055_mean"], fuente)
    coverage_report(df, fuente)
    return df

def clean_static_table(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    df = df.drop(columns=["source", "year"], errors="ignore")  # evita colisiones en el merge del master
    col = next((c for c in ("municipio", "MPO_LEY", "NOM_MUN", "nom_agem") if c in df.columns), None)
    if col is None:
        raise ValueError(f"[{fuente}] sin columna de municipio reconocible. Columnas: {list(df.columns)}")

    df = normalize_municipio(df, col=col, fuente=fuente)
    df = replace_sentinels(df, list(df.select_dtypes("number").columns), fuente=fuente)

    n_dup = int(df.duplicated(subset=["municipio"]).sum())
    if n_dup > 0:
        logger.error(f"[{fuente}] {n_dup} municipios duplicados: se conserva el último. "
                     f"Revisa la ingesta (el master exige 1 fila por municipio)")
        df = df.drop_duplicates(subset=["municipio"], keep="last")
    return df.reset_index(drop=True)

def _parse_wkt(s):
    try:
        if pd.isna(s) or not str(s).strip():
            return None
        return shapely_wkt.loads(s)
    except Exception as e:
        logger.error(f"[municipios_geom] WKT no parseable: {e}")
        return None

def _cve3(x: object) -> str:
    return str(x).strip()[-3:].lstrip("0")

def _detectar_crs(geoms: list) -> str:
    validas = [g for g in geoms if g is not None] # se usa el WKT para detectar CRS, no la geometría proyectada
    if not validas:
        return "EPSG:4326"
    maximo = max(max(abs(v) for v in g.bounds) for g in validas)
    return "EPSG:4326" if maximo <= 360 else "EPSG:6372"

def clean_municipios_geom(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "municipios_geom"
    df = clean_static_table(df, fuente)
    if "geometry_wkt" not in df.columns:
        raise ValueError(f"[{fuente}] no existe 'geometry_wkt'")

    geoms = df["geometry_wkt"].map(_parse_wkt)
    valida = geoms.map(lambda g: g is not None and g.is_valid and not g.is_empty)
    tipo_ok = geoms.map(lambda g: g is not None and g.geom_type in ("Polygon", "MultiPolygon"))
    df["qc_geom_valida"] = valida.astype(bool)
    df["qc_geom_tipo_correcto"] = tipo_ok.astype(bool)
    if (~valida).any():
        logger.error(f"[{fuente}] {int((~valida).sum())} geometrías inválidas o vacías")
    if (~tipo_ok).any():
        logger.error(f"[{fuente}] {int((~tipo_ok).sum())} geometrías que no son Polygon/MultiPolygon")

    # Features numéricas útiles para XGBoost 
    df["geom_area_km2"] = np.nan
    df["geom_centroid_lat"] = np.nan
    df["geom_centroid_lon"] = np.nan
    try:
        geoms_ok = [g if v else None for g, v in zip(geoms, valida)]
        crs_origen = _detectar_crs(geoms_ok)
        logger.info(f"[{fuente}] CRS de la geometría inferido: {crs_origen}")
        gs = gpd.GeoSeries(geoms_ok, index=df.index, crs=crs_origen)
        proj = gs.to_crs("EPSG:6372")  # LCC México (INEGI), área correcta
        df["geom_area_km2"] = proj.area / 1e6
        cen = proj.centroid.to_crs("EPSG:4326")
        df["geom_centroid_lat"] = cen.y
        df["geom_centroid_lon"] = cen.x
    except Exception as e:
        logger.error(f"[{fuente}] no se pudieron calcular área/centroide: {e}")

    if BBOX_AGS is not None:
        try:
            lon_min, lon_max, lat_min, lat_max = (BBOX_AGS[k] for k in ("lon_min", "lon_max", "lat_min", "lat_max"))
        except (KeyError, TypeError, IndexError):
            logger.warning(f"[{fuente}] BBOX_AGS no tiene claves lon_min/lon_max/lat_min/lat_max: se omite validación de bbox")
        else:
            dentro = (
                df["geom_centroid_lon"].between(lon_min, lon_max)
                & df["geom_centroid_lat"].between(lat_min, lat_max)
            )
            df["qc_geom_centroide_en_bbox"] = dentro
            if (~dentro).any():
                logger.error(f"[{fuente}] {int((~dentro).sum())} centroides fuera del bbox -> revisar CRS")

    if INEGI_MUNICIPIOS_AGS is not None and "cve_agem" in df.columns:
        esperado = df["municipio"].map(INEGI_MUNICIPIOS_AGS)
        mismatch = esperado.notna() & (esperado.map(_cve3) != df["cve_agem"].map(_cve3))
        df["qc_geom_cve_agem_ok"] = ~mismatch
        if mismatch.any():
            logger.error(f"[{fuente}] cve_agem no coincide con el catálogo: "
                         f"{df.loc[mismatch, ['municipio', 'cve_agem']].to_dict('records')}")
    return df


def clean_ciclones(df: pd.DataFrame) -> pd.DataFrame:
    fuente = "cenapred_ciclones"
    df = df.drop(columns=["source", "year"], errors="ignore")
    df = normalize_municipio(df, fuente=fuente)

    cols_ciclon = [c for c in df.columns if re.match(r"^l\d+_", c) and any(k in c for k in KEYWORDS_CICLON)]
    if not cols_ciclon:
        raise ValueError(f"[{fuente}] ninguna columna de capa de ciclones. Columnas: {list(df.columns)}")
    otras = [c for c in df.columns if re.match(r"^l\d+_", c) and c not in cols_ciclon]
    if otras:
        logger.info(f"[{fuente}] columnas de otras capas descartadas del master de ciclones: {otras}")

    out = df[["municipio"] + cols_ciclon].copy()

    for c in [c for c in cols_ciclon if pd.api.types.is_datetime64_any_dtype(out[c])]:
        logger.warning(f"[{fuente}] columna datetime '{c}' descartada (XGBoost no la acepta y filtra información futura)")
        out = out.drop(columns=c)

    for c in [c for c in out.columns if c.endswith("_n")]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        if (out[c] < 0).any():
            logger.error(f"[{fuente}] '{c}' con conteos negativos -> NaN")
            out.loc[out[c] < 0, c] = np.nan
        out[c] = out[c].astype("float32")

    n_dup = int(out.duplicated("municipio").sum())
    if n_dup:
        logger.error(f"[{fuente}] {n_dup} municipios duplicados, se conserva el último")
        out = out.drop_duplicates("municipio", keep="last")
    return out.reset_index(drop=True)


FUENTES_DINAMICAS = {
    "nasa_power": (clean_nasa_power, "nasa_power"),
    "chirps": (clean_chirps, "chirps"),
    "openaq": (clean_openaq, "openaq"),
    "maiac_aod": (clean_maiac_aod, "maiac_aod"),
}

FUENTES_ESTATICAS = ["poblacion", "elevacion", "rios", "geometria_municipios", "ciclones"]

LIMPIADORES_ESTATICOS_ESPECIALES = {
    "geometria_municipios": clean_municipios_geom,
    "ciclones": clean_ciclones,
}

# geo_utils.py lee la geometría de inegi.parquet/geometria_municipios.parquet.
MAPEO_RUTAS: dict[str, list[Path]] = {
    "poblacion": [PROCESSED_DIR / "inegi_poblacion.parquet", PROCESSED_DIR / "inegi.parquet" / "poblacion.parquet"],
    "elevacion": [PROCESSED_DIR / "inegi_elevacion.parquet", PROCESSED_DIR / "inegi.parquet" / "elevacion.parquet"],
    "rios": [PROCESSED_DIR / "inegi_rios.parquet", PROCESSED_DIR / "inegi.parquet" / "rios.parquet"],
    "geometria_municipios": [
        PROCESSED_DIR / "inegi.parquet" / "geometria_municipios.parquet",
        PROCESSED_DIR / "inegi_geometria_municipios.parquet",
    ],
    "ciclones": [PROCESSED_DIR / "cenapred.parquet"],
    "maiac_aod": [PROCESSED_DIR / "maiac_aod.parquet"],
}


def _read_processed(nombre: str) -> pd.DataFrame:
    candidatos = MAPEO_RUTAS.get(nombre, [PROCESSED_DIR / f"{nombre}.parquet"])
    path = next((p for p in candidatos if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"{nombre}: no existe ninguna de {[str(p) for p in candidatos]}")
    logger.info(f"{nombre}: leyendo {path}")
    df = ds.dataset(path, format="parquet").to_table().to_pandas() if path.is_dir() else pd.read_parquet(path)
    if "municipio" in df.columns:
        df["municipio"] = df["municipio"].astype(str) 
    return df

def _guardar_master(df: pd.DataFrame, nombre: str, partition_cols: list[str] | None) -> Path:
    """save_parquet particionado no borra archivos previos: escribir a tmp y reemplazar.
    El master anterior se conserva como .bak hasta que el nuevo está en su lugar (swap seguro ante fallos)."""
    out = MASTER_DIR / f"{nombre}.parquet"
    tmp = out.with_name(f"{nombre}.tmp.parquet")
    bak = out.with_name(f"{nombre}.bak.parquet")
    _borrar(tmp)
    _borrar(bak)
    save_parquet(df, tmp, partition_cols=partition_cols, compression=PARQUET_COMPRESSION)
    if out.exists():
        out.rename(bak)
    tmp.rename(out)
    _borrar(bak)
    return out


def clean_dynamic_source(nombre: str, cleaner_fn, source_tag: str) -> pd.DataFrame:
    df = _read_processed(nombre)
    logger.info(f"--- Limpiando {nombre} ({len(df)} filas antes) ---")
    df = cleaner_fn(df).copy()
    if df.duplicated(["date", "municipio"]).any():
        raise ValueError(f"[{nombre}] quedaron duplicados (date, municipio) tras la limpieza")
    df["year"] = df["date"].dt.year
    df["source"] = source_tag
    out = _guardar_master(df, nombre, ["source", "year"])
    logger.info(f"--- {nombre}: {len(df)} filas después -> {out} ---")
    return df


def clean_static_source(nombre: str) -> pd.DataFrame:
    df = _read_processed(nombre)
    logger.info(f"--- Limpiando {nombre} ({len(df)} filas, estática) ---")
    cleaner = LIMPIADORES_ESTATICOS_ESPECIALES.get(nombre)
    df = cleaner(df) if cleaner else clean_static_table(df, nombre)
    out = _guardar_master(df, nombre, None)
    logger.info(f"--- {nombre}: {len(df)} filas -> {out} ---")
    return df


def run_cleaning_pipeline() -> None:
    fallos: dict[str, str] = {}

    for nombre, (fn, tag) in FUENTES_DINAMICAS.items():
        try:
            clean_dynamic_source(nombre, fn, tag)
        except Exception as e:
            fallos[nombre] = f"{type(e).__name__}: {e}"
            logger.error(f"{nombre} falló: {fallos[nombre]}")

    for nombre in FUENTES_ESTATICAS:
        try:
            clean_static_source(nombre)
        except Exception as e:
            fallos[nombre] = f"{type(e).__name__}: {e}"
            logger.error(f"{nombre} falló: {fallos[nombre]}")

    if fallos:
        raise RuntimeError(f"Limpieza incompleta, no corras build_master_table.py. Fallos: {fallos}")
    logger.info("Limpieza completa. data/processed/ no fue modificado; revisa WARNING/ERROR y apunta el master a data/master/.")


if __name__ == "__main__":
    run_cleaning_pipeline()