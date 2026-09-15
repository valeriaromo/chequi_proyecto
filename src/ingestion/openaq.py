from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    OPENAQ_BASE_URL,
    OPENAQ_PARAMETRO_PRINCIPAL,
    BBOX_AGS,
    FECHA_INICIO,
    FECHA_FIN,
    PROCESSED_DIR,
    PARQUET_COMPRESSION,
)
from src.utils.geo_utils import nearest_municipio
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_S = 30

OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY")


def _get(url: str, params: dict) -> dict:
    if not OPENAQ_API_KEY:
        raise RuntimeError(
            "Falta OPENAQ_API_KEY. La API v3 de OpenAQ requiere API key en todas "
        )

    headers = {"X-API-Key": OPENAQ_API_KEY}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT_S)
            if resp.status_code == 401:
                raise RuntimeError(
                    f"OpenAQ respondió 401 Unauthorized para {url}. "
                    "Tu OPENAQ_API_KEY es inválida, expiró, o no se está mandando correctamente"
                )
            if resp.status_code == 429:
                wait = 5 * attempt
                logger.warning(f"OpenAQ: 429 (rate-limit), esperando {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Intento {attempt}/{MAX_RETRIES} falló: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo consultar OpenAQ: {url}")


def _find_stations() -> list[dict]:
    """Descubre estaciones de OpenAQ dentro del bounding box de Aguascalientes."""
    bbox_str = f"{BBOX_AGS['lon_min']},{BBOX_AGS['lat_min']},{BBOX_AGS['lon_max']},{BBOX_AGS['lat_max']}"
    payload = _get(f"{OPENAQ_BASE_URL}/locations", {"bbox": bbox_str, "limit": 100})
    return payload.get("results", [])


def _find_sensor_ids(location_id: int, parametro: str) -> list[int]:
    """
    En v3, "measurements/daily" no cuelga de /locations — cuelga de /sensors.
    Cada location tiene una o más sensores (uno por contaminante/instrumento),
    así que primero hay que listar los sensores de la estación y quedarse
    solo con el/los que midan el parámetro que nos interesa (p.ej. "pm25").
    """
    payload = _get(f"{OPENAQ_BASE_URL}/locations/{location_id}/sensors", {"limit": 100})
    sensors = payload.get("results", [])
    return [
        s["id"] for s in sensors
        if s.get("parameter", {}).get("name") == parametro
    ]


def _fetch_daily_measurements(location_id: int, start: str, end: str) -> pd.DataFrame:
    sensor_ids = _find_sensor_ids(location_id, OPENAQ_PARAMETRO_PRINCIPAL)
    if not sensor_ids:
        return pd.DataFrame(columns=["date", "pm25_mean", "pm25_max", "pm25_p95"])

    datetime_from = f"{start}T00:00:00"
    datetime_to = f"{end}T23:59:59"

    rows = []
    for sensor_id in sensor_ids:
        page = 1
        while True:
            payload = _get(
                f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/days",
                {
                    "datetime_from": datetime_from,
                    "datetime_to": datetime_to,
                    "limit": 1000,
                    "page": page,
                },
            )
            results = payload.get("results", [])
            if not results:
                break
            rows.extend(results)
            if len(results) < 1000:
                break
            page += 1

    if not rows:
        return pd.DataFrame(columns=["date", "pm25_mean", "pm25_max", "pm25_p95"])

    df = pd.json_normalize(rows)
    df = df.rename(columns={
        "period.datetimeFrom.local": "date",
        "value": "pm25_mean",        
        "summary.max": "pm25_max",
        "summary.q98": "pm25_p95",    #
    })
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)

    cols = [c for c in ["date", "pm25_mean", "pm25_max", "pm25_p95"] if c in df.columns]
    return df[cols]


def ingest_openaq(start: date | None = None, end: date | None = None) -> None:
    start_s = (start or FECHA_INICIO).isoformat()
    end_s = (end or FECHA_FIN).isoformat()

    stations = _find_stations()
    logger.info(f"OpenAQ: {len(stations)} estaciones encontradas en Aguascalientes")

    frames = []
    for st in stations:
        loc_id = st["id"]
        coords = st.get("coordinates", {})
        lat, lon = coords.get("latitude"), coords.get("longitude")
        if lat is None or lon is None:
            continue

        municipio = nearest_municipio(lat, lon)
        logger.info(f"OpenAQ: descargando estación {loc_id} -> {municipio}")

        df = _fetch_daily_measurements(loc_id, start_s, end_s)
        if df.empty:
            continue

        df["municipio"] = municipio
        df["station_id"] = loc_id
        frames.append(df)
        time.sleep(0.5)

    if not frames:
        logger.warning("OpenAQ: no se obtuvieron datos, no se guarda nada")
        return

    full_df = pd.concat(frames, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df["source"] = "openaq"
    full_df["year"] = full_df["date"].dt.year

    full_df = (
        full_df.groupby(["date", "municipio", "source", "year"], as_index=False)
        .agg(pm25_mean=("pm25_mean", "mean"),
             pm25_max=("pm25_max", "max"),
             pm25_p95=("pm25_p95", "mean"))
    )

    out_path = PROCESSED_DIR / "openaq.parquet"
    save_parquet(full_df, out_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    logger.info(f"OpenAQ: {len(full_df)} filas guardadas en {out_path}")


if __name__ == "__main__":
    ingest_openaq()