from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    NASA_POWER_BASE_URL,
    NASA_POWER_PARAMS,
    FECHA_INICIO,
    FECHA_FIN,
    PROCESSED_DIR,
    PARQUET_COMPRESSION,
)
from src.utils.geo_utils import municipios_iter
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_S = 60


def _fetch_point(lat: float, lon: float, start: str, end: str) -> dict:
    """Llama a la API de NASA POWER para un solo punto y rango de fechas."""
    params = {
        "parameters": ",".join(NASA_POWER_PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start,
        "end": end,
        "format": "JSON",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(NASA_POWER_BASE_URL, params=params, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"Intento {attempt}/{MAX_RETRIES} falló ({lat},{lon}): {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"No se pudo obtener datos de NASA POWER para ({lat}, {lon})")


def _json_to_dataframe(payload: dict, municipio: str) -> pd.DataFrame:
    params_dict = payload["properties"]["parameter"]

    df = pd.DataFrame({var: pd.Series(values) for var, values in params_dict.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.reset_index()

    df = df.replace(-999, pd.NA)
    df = df.replace(-999.0, pd.NA)

    df["municipio"] = municipio
    df["source"] = "nasa_power"
    df["year"] = df["date"].dt.year

    return df


def ingest_nasa_power(start: str | None = None, end: str | None = None) -> None:
    start = start or FECHA_INICIO.strftime("%Y%m%d")
    end = end or FECHA_FIN.strftime("%Y%m%d")

    frames = []
    for municipio, lat, lon in municipios_iter():
        logger.info(f"NASA POWER: descargando {municipio} ({lat}, {lon})")
        payload = _fetch_point(lat, lon, start, end)
        df = _json_to_dataframe(payload, municipio)
        frames.append(df)
        time.sleep(1) 

    full_df = pd.concat(frames, ignore_index=True)

    out_path = PROCESSED_DIR / "nasa_power.parquet"
    save_parquet(full_df, out_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    logger.info(f"NASA POWER: {len(full_df)} filas guardadas en {out_path}")


if __name__ == "__main__":
    ingest_nasa_power()
