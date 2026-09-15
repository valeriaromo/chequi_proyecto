from __future__ import annotations
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from INEGIpy import MarcoGeoestadistico

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    INEGI_ENTIDAD_AGS,
    INEGI_MUNICIPIOS_AGS,
    OPEN_ELEVATION_URL,
    OVERPASS_URL,
    RIO_SEARCH_RADIUS_M,
    PROCESSED_DIR,
    PARQUET_COMPRESSION,
)
from src.utils.geo_utils import municipios_iter
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3

mg = MarcoGeoestadistico()


def _retry(fn, *args, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except (requests.RequestException, AssertionError) as e:
            logger.warning(f"INEGI intento {attempt}/{MAX_RETRIES} falló: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"INEGI: no se pudo completar la consulta tras {MAX_RETRIES} intentos")


def _ingest_municipios() -> tuple[pd.DataFrame, pd.DataFrame]:
    claves = list(INEGI_MUNICIPIOS_AGS.values())
    gdf = _retry(mg.Municipios, entidades=INEGI_ENTIDAD_AGS, municipios=claves)

    df = pd.DataFrame(gdf.drop(columns="geometry"))
    df["geometry_wkt"] = gdf["geometry"].apply(lambda g: g.wkt)

    clave_a_nombre = {v: k for k, v in INEGI_MUNICIPIOS_AGS.items()}
    if "cve_agem" in df.columns:
        df["municipio"] = df["cve_agem"].astype(str).str.zfill(3).map(clave_a_nombre)
    else:
        logger.warning(
            f"INEGI: no se encontró 'cve_agem' entre {list(df.columns)}. "
            f"Usando 'nom_agem' como respaldo para mapear el nombre."
        )
        df["municipio"] = df.get("nom_agem")

    no_mapeados = df["municipio"].isna().sum()
    if no_mapeados:
        logger.warning(f"INEGI: {no_mapeados} municipios no se pudieron mapear a un nombre conocido")

    df_geom = df[["municipio", "cve_agem", "nom_agem", "geometry_wkt"]].copy()

    cols_pob = [c for c in ["pob", "pob_fem", "pob_mas", "viv"] if c in df.columns]
    if not cols_pob:
        logger.error(f"INEGI: no se encontraron columnas de población entre {list(df.columns)}")
        df_pob = pd.DataFrame(columns=["municipio"])
    else:
        df_pob = df[["municipio"] + cols_pob].copy()
        for c in cols_pob:
            df_pob[c] = pd.to_numeric(df_pob[c], errors="coerce")

    return df_geom, df_pob


def _ingest_elevacion() -> pd.DataFrame:
    puntos = list(municipios_iter())
    locations = [{"latitude": lat, "longitude": lon} for _, lat, lon in puntos]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(OPEN_ELEVATION_URL, json={"locations": locations}, timeout=30)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            break
        except requests.RequestException as e:
            logger.warning(f"Elevación intento {attempt}/{MAX_RETRIES} falló: {e}")
            time.sleep(2 * attempt)
    else:
        logger.error("Elevación: Open-Elevation no respondió tras reintentos")
        return pd.DataFrame(columns=["municipio", "elevacion_msnm"])

    return pd.DataFrame([
        {"municipio": municipio, "elevacion_msnm": r.get("elevation")}
        for (municipio, lat, lon), r in zip(puntos, results)
    ])


OVERPASS_HEADERS = {
    "User-Agent": "chequi-proyecto-ags/1.0 (ingesta de riesgo hidrologico)",
    "Accept": "application/json",
}


def _nearest_river_distance(lat: float, lon: float) -> float | None:
    query = f"""
    [out:json][timeout:25];
    (way["waterway"~"river|stream"](around:{RIO_SEARCH_RADIUS_M},{lat},{lon}););
    out center;
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers=OVERPASS_HEADERS,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 10 * attempt
                logger.warning(f"Overpass: 429 (rate-limit), esperando {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except requests.RequestException as e:
            logger.warning(f"Overpass intento {attempt}/{MAX_RETRIES} falló ({lat},{lon}): {e}")
            time.sleep(2 * attempt)
    else:
        logger.error(f"Overpass: no se pudo consultar tras {MAX_RETRIES} intentos para ({lat},{lon})")
        return None

    if not elements:
        return None

    best = None
    for el in elements:
        center = el.get("center")
        if not center:
            continue
        d = ((center["lat"] - lat) ** 2 + (center["lon"] - lon) ** 2) ** 0.5 * 111_000
        if best is None or d < best:
            best = d
    return best

def _ingest_rios() -> pd.DataFrame:
    rows = []
    for municipio, lat, lon in municipios_iter():
        logger.info(f"Ríos: buscando cercanía para {municipio}")
        dist = _nearest_river_distance(lat, lon)
        rows.append({"municipio": municipio, "distancia_rio_m": dist})
        time.sleep(1)
    return pd.DataFrame(rows)


def ingest_inegi() -> None:
    year = date.today().year

    df_geom, df_pob = _ingest_municipios()

    datasets = {
        "inegi_geometria_municipios": df_geom,
        "inegi_poblacion": df_pob,
        "inegi_elevacion": _ingest_elevacion(),
        "inegi_rios": _ingest_rios(),
    }

    for nombre, df in datasets.items():
        if df.empty:
            logger.warning(f"{nombre}: sin filas, no se guarda")
            continue
        df["source"] = nombre
        df["year"] = year
        out_path = PROCESSED_DIR / f"{nombre}.parquet"
        save_parquet(df, out_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
        logger.info(f"{nombre}: {len(df)} filas guardadas en {out_path}")


if __name__ == "__main__":
    ingest_inegi()