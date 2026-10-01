from __future__ import annotations
import json
import logging
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (BBOX_AGS,CENAPRED_ARCGIS_MAPSERVER,MUNICIPIOS_AGS,PARQUET_COMPRESSION,PROCESSED_DIR,)
from src.utils.geo_utils import load_municipios_geom
from src.utils.io_utils import save_parquet
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MAX_RETRIES = 3
TIMEOUT_S = 30
ID_CHUNK = 200  # objectIds por request

HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ""(KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"),"Accept": "application/json",}

KEYWORDS_RIESGO = ["desastre", "riesgo", "peligro", "emergencia", "declaratoria"]
COLS_MUNICIPIO = {"municipio", "nom_mun", "nom_municipio", "nombre_municipio", "nomgeo", "mun"}
COLS_ENTIDAD = {"entidad", "estado", "nom_ent", "nom_estado", "noment", "nom_entidad", "cve_ent"}
RE_COLS_EXCLUIDAS = re.compile(r"^(objectid|fid|gid|id|shape.*|cve.*|cvegeo|index_right)$", re.I)

def _normalizar(s: object) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()

def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalizar(s)).strip("_")[:40]

def _get_json(url: str, params: dict | None = None) -> dict:
    params = {**(params or {}), "f": "json"}
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
        except requests.RequestException as e:
            last_err = e
            logger.warning(f"CENAPRED intento {attempt}/{MAX_RETRIES} falló ({url}): {e}")
            time.sleep(2 * attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = RuntimeError(f"HTTP {resp.status_code}")
            logger.warning(f"CENAPRED: HTTP {resp.status_code}, reintento {attempt}/{MAX_RETRIES}")
            time.sleep(2 * attempt)
            continue
        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code} no recuperable en {url}: {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Respuesta no JSON en {url}") from e
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"ArcGIS error en {url}: {data['error'].get('message')} {data['error'].get('details')}")
        return data
    raise RuntimeError(f"CENAPRED: no se pudo consultar {url}: {last_err}")


def _discover_layers() -> list[dict]:
    payload = _get_json(CENAPRED_ARCGIS_MAPSERVER)
    layers = payload.get("layers", [])
    relevantes = [
        l for l in layers
        if not l.get("subLayerIds")
        and any(kw in l.get("name", "").lower() for kw in KEYWORDS_RIESGO)
    ]
    logger.info(f"CENAPRED: {len(layers)} capas totales, {len(relevantes)} relevantes a riesgo")
    for l in relevantes:
        logger.info(f"  -> [{l['id']}] {l['name']}")
    return relevantes

def _anillo_horario(ring: list) -> bool:
    area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]))
    return area < 0 

def _esri_to_shapely(g: dict | None):
    if not g:
        return None
    if "x" in g and "y" in g:
        try:
            return Point(float(g["x"]), float(g["y"]))
        except (TypeError, ValueError):
            return None
    if "points" in g:
        return MultiPoint([p[:2] for p in g["points"]]) if g["points"] else None
    if "paths" in g:
        paths = [[p[:2] for p in path] for path in g["paths"] if len(path) >= 2]
        return MultiLineString(paths) if paths else None
    if "rings" in g:
        rings = [[tuple(p[:2]) for p in r] for r in g["rings"] if len(r) >= 4]
        exteriores = [r for r in rings if _anillo_horario(r)]
        huecos = [r for r in rings if not _anillo_horario(r)]
        if not exteriores:
            exteriores, huecos = huecos, []
        polys = [Polygon(r) for r in exteriores]
        for h in huecos:
            punto = Polygon(h).representative_point()
            for i, p in enumerate(polys):
                if p.contains(punto):
                    polys[i] = Polygon(p.exterior.coords, list(p.interiors) + [h])
                    break
        if not polys:
            return None
        geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        return geom if geom.is_valid else geom.buffer(0)
    return None


def _query_layer(layer_id: int) -> gpd.GeoDataFrame:
    base = f"{CENAPRED_ARCGIS_MAPSERVER}/{layer_id}"
    info = _get_json(base)
    geometry_type = info.get("geometryType")
    campos_fecha = {f["name"] for f in info.get("fields", []) if f.get("type") == "esriFieldTypeDate"}
    filtro: dict = {"where": "1=1"}
    if geometry_type:
        filtro.update({
            "geometry": json.dumps({
                "xmin": BBOX_AGS["lon_min"], "ymin": BBOX_AGS["lat_min"],
                "xmax": BBOX_AGS["lon_max"], "ymax": BBOX_AGS["lat_max"],
                "spatialReference": {"wkid": 4326},
            }),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
        })

    ids = _get_json(f"{base}/query", {**filtro, "returnIdsOnly": "true"}).get("objectIds") or []
    if not ids:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    features: list[dict] = []
    for i in range(0, len(ids), ID_CHUNK):
        chunk = ids[i:i + ID_CHUNK]
        data = _get_json(f"{base}/query", {
            "objectIds": ",".join(map(str, chunk)),
            "outFields": "*",
            "returnGeometry": "true" if geometry_type else "false",
            "outSR": "4326",
        })
        features.extend(data.get("features", []))

    df = pd.DataFrame([f.get("attributes", {}) for f in features])
    for c in campos_fecha & set(df.columns):
        df[c] = pd.to_datetime(df[c], unit="ms", errors="coerce")
    geoms = [_esri_to_shapely(f.get("geometry")) for f in features]
    return gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")


def _asignar_por_atributo(gdf: gpd.GeoDataFrame) -> pd.DataFrame | None:
    col_mun = next((c for c in gdf.columns if str(c).lower() in COLS_MUNICIPIO), None)
    if col_mun is None:
        return None
    df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))

    col_ent = next((c for c in df.columns if str(c).lower() in COLS_ENTIDAD), None)
    if col_ent is not None:
        v = df[col_ent].map(_normalizar)
        df = df[v.isin({"aguascalientes", "ags", "1", "01"})]

    cat = {_normalizar(m): m for m in MUNICIPIOS_AGS}
    df = df.assign(municipio=df[col_mun].map(lambda x: cat.get(_normalizar(x))))
    df = df.dropna(subset=["municipio"])
    return df if not df.empty else None


def _agregar_capa(gdf: gpd.GeoDataFrame, layer_id: int, layer_name: str, mun_geom: gpd.GeoDataFrame) -> pd.DataFrame | None:
    prefijo = f"l{layer_id}_{_slug(layer_name)}"
    col_n = f"{prefijo}_n"
    todos = list(MUNICIPIOS_AGS)

    if gdf.empty:
        return pd.DataFrame({"municipio": todos, col_n: 0})

    joined = _asignar_por_atributo(gdf)
    if joined is None:
        tiene_geom = "geometry" in gdf and gdf.geometry.notna().any()
        if not tiene_geom:
            logger.warning(f"  -> Capa [{layer_id}] sin geometría ni columna de municipio reconocible: se omite")
            return None
        g = gdf[gdf.geometry.notna()]
        g = g.rename(columns={c: f"{c}_attr" for c in g.columns if c in ("municipio", "index_right")})
        joined = gpd.sjoin(g, mun_geom[["municipio", "geometry"]], how="inner", predicate="intersects")
        joined = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))

    joined = joined[joined["municipio"].isin(MUNICIPIOS_AGS)]
    if joined.empty:
        return pd.DataFrame({"municipio": todos, col_n: 0})

    cols_val = [
        c for c in joined.select_dtypes(include=["number", "datetime"]).columns
        if not RE_COLS_EXCLUIDAS.match(str(c))
    ]
    agg = joined.groupby("municipio").size().rename(col_n).to_frame()
    if cols_val:
        maxs = joined.groupby("municipio")[cols_val].max()
        maxs.columns = [f"{prefijo}_{_slug(str(c))}_max" for c in maxs.columns]
        agg = agg.join(maxs)

    out = agg.reindex(todos).rename_axis("municipio").reset_index()
    out[col_n] = out[col_n].fillna(0).astype("int32")
    return out


def ingest_cenapred() -> None:
    try:
        layers = _discover_layers()
    except RuntimeError as e:
        raise RuntimeError(f"CENAPRED: error consultando MapServer: {e}") from e
    if not layers:
        raise RuntimeError("CENAPRED: no se encontraron capas con las palabras clave")

    try:
        mun_geom = load_municipios_geom()
    except FileNotFoundError as e:
        raise RuntimeError("Falta geometria_municipios.parquet: corre inegi.py antes de cenapred.py") from e

    frames: list[pd.DataFrame] = []
    fallidas: list[int] = []
    for layer in layers:
        layer_id, layer_name = layer["id"], layer["name"]
        logger.info(f"CENAPRED: procesando capa [{layer_id}] {layer_name}")
        try:
            gdf = _query_layer(layer_id)
            logger.info(f"  -> {len(gdf)} features; columnas: {[c for c in gdf.columns if c != 'geometry']}")
            agg = _agregar_capa(gdf, layer_id, layer_name, mun_geom)
        except RuntimeError as e:
            fallidas.append(layer_id)
            logger.error(f"CENAPRED: capa {layer_id} falló, se continúa: {e}")
            continue
        if agg is not None:
            frames.append(agg)
        time.sleep(0.3)

    if not frames:
        raise RuntimeError(f"CENAPRED: ninguna capa produjo datos (fallidas: {fallidas}), no se guarda nada")

    wide = pd.DataFrame({"municipio": list(MUNICIPIOS_AGS)})
    for df in frames:
        n = len(wide)
        wide = wide.merge(df, on="municipio", how="left")
        assert len(wide) == n, "El merge de capas CENAPRED alteró el número de filas"
    assert wide["municipio"].is_unique and len(wide) == len(MUNICIPIOS_AGS)

    wide["source"] = "cenapred"
    wide["year"] = pd.Timestamp.today().year
    out_path = PROCESSED_DIR / "cenapred.parquet"
    tmp_path = out_path.with_name("cenapred.tmp.parquet")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    save_parquet(wide, tmp_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    if out_path.exists():
        shutil.rmtree(out_path)
    tmp_path.rename(out_path)
    logger.info(f"CENAPRED: {len(wide)} filas x {wide.shape[1]} columnas guardadas en {out_path}")

    if fallidas:
        raise RuntimeError(f"CENAPRED: datos parciales guardados; fallaron las capas {fallidas}")


if __name__ == "__main__":
    ingest_cenapred()