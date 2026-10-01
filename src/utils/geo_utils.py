from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MUNICIPIOS_AGS, BBOX_AGS, PROCESSED_DIR

_geom_cache: gpd.GeoDataFrame | None = None


def bbox_window():
    return (
        BBOX_AGS["lon_min"],
        BBOX_AGS["lat_min"],
        BBOX_AGS["lon_max"],
        BBOX_AGS["lat_max"],
    )


def municipios_iter():
    for nombre, coords in MUNICIPIOS_AGS.items():
        yield nombre, coords["lat"], coords["lon"]


def nearest_municipio(lat: float, lon: float) -> str:
    best_name, best_dist = None, float("inf")
    for nombre, m_lat, m_lon in municipios_iter():
        dist = (lat - m_lat) ** 2 + (lon - m_lon) ** 2
        if dist < best_dist:
            best_name, best_dist = nombre, dist
    return best_name


def load_municipios_geom() -> gpd.GeoDataFrame:
    global _geom_cache
    if _geom_cache is not None:
        return _geom_cache

    path = PROCESSED_DIR / "inegi.parquet"/ "geometria_municipios.parquet"
    df = pd.read_parquet(path)

    if "year" in df.columns:
        df = df.sort_values("year").drop_duplicates(subset="municipio", keep="last")

    df = df.dropna(subset=["municipio", "geometry_wkt"]).copy()
    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    _geom_cache = gdf
    return gdf


def municipios_iter_geom():
    gdf = load_municipios_geom()
    centroids = gdf.geometry.centroid
    for municipio, geom_centroid in zip(gdf["municipio"], centroids):
        yield municipio, geom_centroid.y, geom_centroid.x  # lat, lon


def municipio_from_point(lat: float, lon: float) -> str | None:
    gdf = load_municipios_geom()
    pt = Point(lon, lat)
    match = gdf[gdf.geometry.contains(pt)]
    if not match.empty:
        return match.iloc[0]["municipio"]
    return nearest_municipio(lat, lon)