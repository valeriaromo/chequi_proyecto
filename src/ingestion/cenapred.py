from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CENAPRED_ARCGIS_MAPSERVER,
    CENAPRED_ESTADO_FILTRO,
    PROCESSED_DIR,
    PARQUET_COMPRESSION,
)
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_S = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

KEYWORDS_RIESGO = ["desastre", "riesgo", "peligro", "emergencia", "declaratoria"]
CAMPOS_ESTADO_CANDIDATOS = ["EDO_LEY", "ESTADO", "NOM_ENT", "NOMGEO", "ENTIDAD"]


def _get_json(url: str, params: dict | None = None) -> dict:
    params = params or {}
    params["f"] = "json"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.warning(f"CENAPRED intento {attempt}/{MAX_RETRIES} falló ({url}): {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"CENAPRED: no se pudo consultar {url}")


def _discover_layers() -> list[dict]:
    """Lista todas las capas del MapServer y regresa solo las relacionadas a riesgo."""
    payload = _get_json(CENAPRED_ARCGIS_MAPSERVER)
    layers = payload.get("layers", [])
    relevantes = [
        l for l in layers
        if any(kw in l.get("name", "").lower() for kw in KEYWORDS_RIESGO)
    ]
    logger.info(f"CENAPRED: {len(layers)} capas totales, {len(relevantes)} relevantes a riesgo")
    for l in relevantes:
        logger.info(f"  -> [{l['id']}] {l['name']}")
    return relevantes


def _query_layer_estado(layer_id: int, campo: str, estado: str) -> list[dict]:
    url = f"{CENAPRED_ARCGIS_MAPSERVER}/{layer_id}/query"
    where = f"UPPER({campo}) LIKE UPPER('%{estado}%')"
    payload = _get_json(url, {
        "where": where,
        "outFields": "*",
        "returnGeometry": "false",
    })
    return payload.get("features", [])


def _query_layer_con_deteccion_campo(layer_id: int, estado: str) -> list[dict]:
    for campo in CAMPOS_ESTADO_CANDIDATOS:
        try:
            features = _query_layer_estado(layer_id, campo, estado)
        except RuntimeError:
            continue

        if features:
            logger.info(f"  -> campo de estado detectado: {campo} ({len(features)} features)")
            return features

    logger.warning(
        f"CENAPRED: ningún campo candidato ({CAMPOS_ESTADO_CANDIDATOS}) devolvió "
        f"resultados para '{estado}' en la capa {layer_id}. Puede que el nombre del "
        f"campo no esté en la lista de candidatos, o que el valor de estado en la "
        f"capa no coincida textualmente con '{estado}' (p. ej. abreviado o con "
        f"clave numérica en vez de nombre) — revisa manualmente los campos de la "
        f"capa antes de descartarla."
    )
    return []


def ingest_cenapred() -> None:
    try:
        layers = _discover_layers()
    except RuntimeError as e:
        logger.error(f"CENAPRED: no se pudo ni siquiera listar las capas del MapServer: {e}")
        logger.error("Revisa si la URL de config.settings.CENAPRED_ARCGIS_MAPSERVER sigue vigente.")
        return

    if not layers:
        logger.warning("CENAPRED: no se encontraron capas relevantes con las keywords configuradas")
        return

    frames = []
    for layer in layers:
        layer_id, layer_name = layer["id"], layer["name"]
        logger.info(f"CENAPRED: consultando capa [{layer_id}] {layer_name}")

        try:
            features_ags = _query_layer_con_deteccion_campo(layer_id, CENAPRED_ESTADO_FILTRO)
        except RuntimeError as e:
            logger.error(f"CENAPRED: capa {layer_name} falló: {e}")
            continue

        logger.info(f"  -> {len(features_ags)} features en {CENAPRED_ESTADO_FILTRO}")

        if not features_ags:
            continue

        df = pd.json_normalize([f["attributes"] for f in features_ags])
        df["layer_id"] = layer_id
        df["layer_name"] = layer_name
        frames.append(df)
        time.sleep(0.5)

    if not frames:
        logger.warning("CENAPRED: no se obtuvieron features para Aguascalientes en ninguna capa")
        return

    full_df = pd.concat(frames, ignore_index=True, sort=False)
    full_df["source"] = "cenapred"
    full_df["year"] = pd.Timestamp.today().year

    out_path = PROCESSED_DIR / "cenapred.parquet"
    save_parquet(full_df, out_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    logger.info(f"CENAPRED: {len(full_df)} filas guardadas en {out_path}")


if __name__ == "__main__":
    ingest_cenapred()