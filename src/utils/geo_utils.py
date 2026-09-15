"""
Utilidades geoespaciales ligeras.
No se usa PostGIS aquí a propósito: para el volumen de este proyecto
(11 municipios, tabla maestra de pocos MB), un bounding box + centroides
es más que suficiente. PostGIS se reserva para consultas geográficas
más avanzadas en fases posteriores (uso de suelo, polígonos reales, etc.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import MUNICIPIOS_AGS, BBOX_AGS


def bbox_window():
    """Devuelve el bounding box de Aguascalientes como tupla (minx, miny, maxx, maxy)."""
    return (
        BBOX_AGS["lon_min"],
        BBOX_AGS["lat_min"],
        BBOX_AGS["lon_max"],
        BBOX_AGS["lat_max"],
    )


def municipios_iter():
    """Itera (nombre_municipio, lat, lon) para todos los municipios configurados."""
    for nombre, coords in MUNICIPIOS_AGS.items():
        yield nombre, coords["lat"], coords["lon"]


def nearest_municipio(lat: float, lon: float) -> str:
    """
    Devuelve el municipio cuyo centroide está más cerca de (lat, lon).
    Útil para asignar estaciones de OpenAQ a un municipio.
    """
    best_name, best_dist = None, float("inf")
    for nombre, m_lat, m_lon in municipios_iter():
        dist = (lat - m_lat) ** 2 + (lon - m_lon) ** 2  # distancia euclidiana simple, suficiente a esta escala
        if dist < best_dist:
            best_name, best_dist = nombre, dist
    return best_name
