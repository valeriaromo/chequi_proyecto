from __future__ import annotations
import logging
import os
import shutil
import sys
import time
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (OPENAQ_BASE_URL,OPENAQ_PARAMETRO_PRINCIPAL,BBOX_AGS,MUNICIPIOS_AGS,FECHA_INICIO,FECHA_FIN,PROCESSED_DIR,PARQUET_COMPRESSION,)
from src.utils.geo_utils import load_municipios_geom
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_S = 30
MAX_PAGES = 100
PM25_MAX_FISICO = 1000.0  # µg/m³; fuera de [0, 1000] se trata como sentinela/error

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")

RENAME_MAP = {"period.datetimeFrom.local": "date","value": "pm25_mean","summary.max": "pm25_max",
"summary.q98": "pm25_q98",  # OpenAQ no tienene q95 entonces se usa ese    
}
OUT_COLS = ["date", "pm25_mean", "pm25_max", "pm25_q98"]
VALUE_COLS = ["pm25_mean", "pm25_max", "pm25_q98"]


def _get(url: str, params: dict) -> dict:
    if not OPENAQ_API_KEY:
        raise RuntimeError("Falta OPENAQ_API_KEY: la API v3 de OpenAQ la exige en todas las peticiones.")
    headers = {"X-API-Key": OPENAQ_API_KEY}
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"Intento {attempt}/{MAX_RETRIES} falló: {e}")
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 401:
            raise RuntimeError(f"OpenAQ 401 Unauthorized para {url}: OPENAQ_API_KEY inválida o expirada.")
        if resp.status_code == 404:
            return {"results": []}
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 5 * attempt
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            logger.warning(f"OpenAQ: HTTP {resp.status_code}, esperando {wait}s...")
            time.sleep(wait)
            continue
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"OpenAQ error no recuperable en {url}: {e}") from e
        return resp.json()

    raise RuntimeError(f"No se pudo consultar OpenAQ: {url} ({last_err})")


def _paginate(url: str, params: dict, limit: int) -> list[dict]:
    out: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        payload = _get(url, {**params, "limit": limit, "page": page})
        results = payload.get("results", [])
        out.extend(results)
        if len(results) < limit:
            break
    return out


def _find_stations() -> list[dict]:
    bbox_str = f"{BBOX_AGS['lon_min']},{BBOX_AGS['lat_min']},{BBOX_AGS['lon_max']},{BBOX_AGS['lat_max']}"
    stations = _paginate(f"{OPENAQ_BASE_URL}/locations", {"bbox": bbox_str}, limit=100)
    unicas = {st["id"]: st for st in stations if "id" in st}
    return list(unicas.values())


def _find_sensor_ids(location_id: int, parametro: str) -> list[int]:
    sensors = _paginate(f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors", {}, limit=100)
    objetivo = parametro.lower()
    return [
        s["id"] for s in sensors
        if str((s.get("parameter") or {}).get("name", "")).lower() == objetivo
    ]


def _fetch_daily_measurements(location_id: int, start: str, end: str) -> pd.DataFrame:
    sensor_ids = _find_sensor_ids(location_id, OPENAQ_PARAMETRO_PRINCIPAL)
    if not sensor_ids:
        return pd.DataFrame(columns=OUT_COLS)

    rows: list[dict] = []
    for sensor_id in sensor_ids:
        rows.extend(
            _paginate(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/days",
                {"datetime_from": f"{start}T00:00:00", "datetime_to": f"{end}T23:59:59"},
                limit=1000,
            )
        )
    if not rows:
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.json_normalize(rows).rename(columns=RENAME_MAP).reindex(columns=OUT_COLS)
    df["date"] = df["date"].astype(str).str[:10]  

    for col in VALUE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[(df[col] < 0) | (df[col] > PM25_MAX_FISICO), col] = np.nan

    return df.dropna(subset=["pm25_mean"])


def _municipio_estricto(lat: float, lon: float, gdf) -> str | None: #usar puntos en estados cercanos si no se encuentra el municipio
    match = gdf[gdf.geometry.covers(Point(lon, lat))]
    return None if match.empty else str(match.iloc[0]["municipio"])


def _merge_con_existente(nuevo: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return nuevo
    existente = pd.read_parquet(path)
    existente["municipio"] = existente["municipio"].astype(str)
    existente["date"] = pd.to_datetime(existente["date"])
    comb = pd.concat([existente[nuevo.columns], nuevo], ignore_index=True)
    comb = comb.drop_duplicates(subset=["date", "municipio"], keep="last")
    comb["source"] = "openaq"
    comb["year"] = comb["date"].dt.year
    return comb


def ingest_openaq(start: date | None = None, end: date | None = None) -> None:
    start_s = (start or FECHA_INICIO).isoformat()
    end_s = (end or FECHA_FIN).isoformat()
    gdf = load_municipios_geom()
    stations = _find_stations()
    logger.info(f"OpenAQ: {len(stations)} estaciones en el bbox")

    frames: list[pd.DataFrame] = []
    fuera, fallidas = 0, 0
    for st in stations:
        loc_id = st["id"]
        coords = st.get("coordinates") or {}
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue
        municipio = _municipio_estricto(lat, lon, gdf)
        if municipio is None or municipio not in MUNICIPIOS_AGS:
            fuera += 1
            logger.info(f"OpenAQ: estación {loc_id} fuera de Aguascalientes o municipio no catalogado, se omite")
            continue
        logger.info(f"OpenAQ: descargando estación {loc_id} -> {municipio}")
        try:
            df = _fetch_daily_measurements(loc_id, start_s, end_s)
        except RuntimeError as e:
            if "401" in str(e) or "OPENAQ_API_KEY" in str(e):
                raise
            fallidas += 1
            logger.error(f"OpenAQ: estación {loc_id} falló, se continúa: {e}")
            continue
        if df.empty:
            continue

        df["municipio"] = municipio
        frames.append(df)
        time.sleep(0.5)

    logger.info(f"OpenAQ: {len(frames)} con datos, {fuera} fuera de estado, {fallidas} fallidas")
    if not frames:
        logger.warning("OpenAQ: no se obtuvieron datos, no se guarda nada")
        return

    full_df = pd.concat(frames, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = (
        full_df.groupby(["date", "municipio"], as_index=False)
        .agg(pm25_mean=("pm25_mean", "mean"),
             pm25_max=("pm25_max", "max"),
             pm25_q98=("pm25_q98", "mean"))
    )
    full_df["source"] = "openaq"
    full_df["year"] = full_df["date"].dt.year
    out_path = PROCESSED_DIR / "openaq.parquet"
    full_df = _merge_con_existente(full_df, out_path)
    tmp_path = out_path.with_name("openaq.tmp.parquet")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    save_parquet(full_df, tmp_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    logger.info(f"OpenAQ: {len(full_df)} filas guardadas en {out_path}")

if __name__ == "__main__":
    ingest_openaq()