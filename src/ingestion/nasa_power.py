from __future__ import annotations
import logging
import shutil
import sys
import time
from datetime import date
from pathlib import Path
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (NASA_POWER_BASE_URL,NASA_POWER_PARAMS,MUNICIPIOS_AGS,FECHA_INICIO,FECHA_FIN,PROCESSED_DIR,PARQUET_COMPRESSION)
from src.utils.geo_utils import municipios_iter_geom
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_S = 60
SENTINELA = -999


def _to_yyyymmdd(value: date | str) -> str:
    if isinstance(value, str):
        return value.replace("-", "")
    return value.strftime("%Y%m%d")


def _fetch_point(lat: float, lon: float, start: str, end: str) -> dict:
    params = {"parameters": ",".join(NASA_POWER_PARAMS),"community": "AG","longitude": lon,
        "latitude": lat,"start": start,"end": end,"format": "JSON",}
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(NASA_POWER_BASE_URL, params=params, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"Intento {attempt}/{MAX_RETRIES} falló ({lat:.4f},{lon:.4f}): {e}")
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 5 * attempt
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            logger.warning(f"NASA POWER: HTTP {resp.status_code}, esperando {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code >= 400: # no se recupera con reintentos
            raise RuntimeError(f"NASA POWER error no recuperable {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    raise RuntimeError(f"No se pudo obtener datos de NASA POWER para ({lat}, {lon}): {last_err}")


def _json_to_dataframe(payload: dict, municipio: str) -> pd.DataFrame:
    try:
        params_dict = payload["properties"]["parameter"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"Respuesta de NASA POWER sin 'properties.parameter' para {municipio}") from e

    fill = (payload.get("header") or {}).get("fill_value", SENTINELA)
    df = pd.DataFrame({var: pd.Series(values) for var, values in params_dict.items()})
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask((df == fill) | (df == SENTINELA))
    df = df.reindex(columns=list(NASA_POWER_PARAMS))  # esquema estable aunque falte una variable
    df = df.dropna(how="all")  # es para eliminar días sin datos válidos en ninguna variable
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"
    df = df.reset_index()
    df["municipio"] = municipio
    df["source"] = "nasa_power"
    df["year"] = df["date"].dt.year
    return df


def _merge_con_existente(nuevo: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return nuevo
    existente = pd.read_parquet(path)
    existente["municipio"] = existente["municipio"].astype(str)
    existente["date"] = pd.to_datetime(existente["date"])
    cols = [c for c in nuevo.columns if c in existente.columns]
    comb = pd.concat([existente[cols], nuevo], ignore_index=True)
    comb = comb.drop_duplicates(subset=["date", "municipio"], keep="last")
    comb["source"] = "nasa_power"
    comb["year"] = comb["date"].dt.year
    return comb


def ingest_nasa_power(start: date | str | None = None, end: date | str | None = None) -> None:
    start_s = _to_yyyymmdd(start or FECHA_INICIO)
    end_s = _to_yyyymmdd(end or FECHA_FIN)
    if start_s > end_s:
        raise ValueError(f"Rango inválido: {start_s} > {end_s}")
    try:
        centroides = list(municipios_iter_geom())
    except FileNotFoundError as e:
        raise RuntimeError(
            "Falta geometria_municipios.parquet: corre inegi.py antes de nasa_power.py"
        ) from e

    frames: list[pd.DataFrame] = []
    fallidos: list[str] = []
    for municipio, lat, lon in centroides:
        if municipio not in MUNICIPIOS_AGS:
            logger.warning(f"NASA POWER: '{municipio}' no está en MUNICIPIOS_AGS, se omite")
            continue
        logger.info(f"NASA POWER: descargando {municipio} ({lat:.4f}, {lon:.4f})")
        try:
            payload = _fetch_point(lat, lon, start_s, end_s)
            df = _json_to_dataframe(payload, municipio)
        except (RuntimeError, ValueError) as e:
            fallidos.append(municipio)
            logger.error(f"NASA POWER: {municipio} falló, se continúa: {e}")
            continue
        if df.empty:
            logger.warning(f"NASA POWER: {municipio} sin datos válidos en {start_s}-{end_s}")
            continue
        frames.append(df)
        time.sleep(1)

    cubiertos = {f["municipio"].iloc[0] for f in frames}
    faltantes = sorted(set(MUNICIPIOS_AGS) - cubiertos)
    if faltantes:
        logger.warning(f"NASA POWER: municipios sin datos: {faltantes}")

    if not frames:
        raise RuntimeError("NASA POWER: no se obtuvieron datos de ningún municipio, no se guarda nada")

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["date", "municipio"], keep="last")
    out_path = PROCESSED_DIR / "nasa_power.parquet"
    full_df = _merge_con_existente(full_df, out_path)
    tmp_path = out_path.with_name("nasa_power.tmp.parquet")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    save_parquet(full_df, tmp_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    logger.info(f"NASA POWER: {len(full_df)} filas guardadas en {out_path}")
    if fallidos:
        raise RuntimeError(f"NASA POWER: datos parciales guardados; fallaron {fallidos}")


if __name__ == "__main__":
    ingest_nasa_power()