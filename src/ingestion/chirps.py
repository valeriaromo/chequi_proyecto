from __future__ import annotations

import argparse
import gzip
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.io import MemoryFile
from rasterio.windows import from_bounds

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (
    CHIRPS_DAILY_COG_TEMPLATE,
    PROCESSED_DIR,
    PARQUET_COMPRESSION,
)
from src.utils.geo_utils import bbox_window, municipios_iter
from src.utils.io_utils import save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


DEFAULT_START = date(2020, 1, 1)
DEFAULT_END = date.today() - timedelta(days=1)

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
    "VSI_CACHE": "TRUE",
    "GDAL_HTTP_USERAGENT": HTTP_HEADERS["User-Agent"],
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
}

MAX_RETRIES = 3
REQUEST_TIMEOUT_S = 30

GZIP_MAGIC = b"\x1f\x8b"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*")


def _daily_url(d: date, gz: bool = True) -> str:
    url = CHIRPS_DAILY_COG_TEMPLATE.format(year=d.year, month=d.month, day=d.day)
    if gz:
        return url
    return url[:-3] if url.endswith(".gz") else url


def _maybe_gunzip(content: bytes) -> bytes:
    if content[:2] == GZIP_MAGIC:
        return gzip.decompress(content)
    return content


def _looks_like_tiff(content: bytes) -> bool:
    return content[:4] in TIFF_MAGICS


def _progress_bar(i: int, total: int, prefix: str = "", width: int = 30) -> str:
    """Barra de progreso simple sin dependencias externas."""
    if total <= 0:
        return f"{prefix} [{i}]"
    frac = i / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    return f"{prefix} |{bar}| {i}/{total} ({frac*100:5.1f}%)"


def diagnose_date(d: date) -> None:
    url = _daily_url(d, gz=True)
    print("=" * 70)
    print(f"DIAGNÓSTICO CHIRPS — fecha {d.isoformat()}")
    print(f"URL: {url}")
    print("=" * 70)

    print("\n[1/4] Probando HTTP directo con 'requests' (HEAD)...")
    try:
        resp = requests.head(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_S, allow_redirects=True)
        print(f"      Status: {resp.status_code}")
        print(f"      Content-Type: {resp.headers.get('Content-Type')}")
        print(f"      Content-Length: {resp.headers.get('Content-Length')}")
        if resp.status_code == 403:
            print("      -> 403 FORBIDDEN: el servidor está bloqueando la solicitud.")
        elif resp.status_code == 404:
            print("      -> 404 con extensión .gz: probable hueco real del catálogo.")
        elif resp.status_code == 429:
            print("      -> 429 TOO MANY REQUESTS: rate-limited, baja la frecuencia.")
        elif resp.status_code >= 500:
            print("      -> Error del lado del servidor, reintentar más tarde.")
        elif resp.status_code == 200:
            print("      -> OK. El dominio responde y el archivo .tif.gz existe.")
    except requests.exceptions.SSLError as e:
        print(f"      -> ERROR SSL: {e}")
    except requests.exceptions.ConnectTimeout as e:
        print(f"      -> TIMEOUT DE CONEXIÓN: {e}")
        print("      -> Tu red/sandbox NO tiene salida a este dominio.")
    except requests.exceptions.ConnectionError as e:
        print(f"      -> ERROR DE CONEXIÓN: {e}")
    except requests.exceptions.RequestException as e:
        print(f"      -> ERROR GENÉRICO: {e}")

    print("\n[2/4] Probando descarga de bytes con 'requests' (GET, con headers)...")
    content = None
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_S)
        print(f"      Status: {resp.status_code}, bytes recibidos: {len(resp.content)}")
        if resp.status_code == 200 and resp.content[:2] == GZIP_MAGIC:
            print("      -> OK. gzip válido. Se descomprimirá.")
            content = resp.content
        elif resp.status_code == 200 and _looks_like_tiff(resp.content):
            print("      -> OK. TIFF sin comprimir (raro, pero se acepta).")
            content = resp.content
        elif resp.status_code == 200:
            print(f"      -> 200 pero contenido NO gzip/TIFF (primeros bytes: {resp.content[:20]!r}).")
        else:
            print(f"      -> No se pudo descargar (status {resp.status_code}).")
    except requests.exceptions.RequestException as e:
        print(f"      -> ERROR: {e}")

    print("\n[3/4] Descomprimir (si aplica) y abrir con rasterio (MemoryFile)...")
    if content:
        try:
            tif_bytes = _maybe_gunzip(content)
            if not _looks_like_tiff(tif_bytes):
                print(f"      -> ERROR: tras descomprimir no parece TIFF ({tif_bytes[:20]!r}).")
            else:
                with MemoryFile(tif_bytes) as memfile:
                    with memfile.open() as src:
                        print(f"      -> OK. Raster válido: {src.width}x{src.height}, "
                              f"CRS={src.crs}, nodata={src.nodata}")
        except Exception as e:
            print(f"      -> ERROR al descomprimir/abrir con rasterio: {e}")
    else:
        print("      -> Se omite (no hay contenido del paso 2).")

    print("\n[4/4] Probando acceso remoto vía GDAL /vsigzip//vsicurl/ ...")
    vsi_path = f"/vsigzip//vsicurl/{url}"
    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(vsi_path) as src:
                print(f"      -> OK. /vsigzip//vsicurl/ funciona: {src.width}x{src.height}, CRS={src.crs}")
    except rasterio.errors.RasterioIOError as e:
        print(f"      -> ERROR RasterioIOError: {e}")
        print("      -> Si 2/3 OK pero este falla, usa el fallback 'download'.")
    except Exception as e:
        print(f"      -> ERROR inesperado: {e}")

    print("\n" + "=" * 70)
    print("RESUMEN: revisa cuál fue el primer paso en fallar arriba.")
    print("=" * 70)


def _download_bytes(d: date) -> bytes | None:
    for gz in (True, False):
        url = _daily_url(d, gz=gz)
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"      · GET {url} (intento {attempt}/{MAX_RETRIES})")
                resp = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_S)
                print(f"        status={resp.status_code} bytes={len(resp.content)}")
                if resp.status_code == 404:
                    break
                if resp.status_code == 429:
                    wait = 5 * attempt
                    print(f"        429 rate-limit, esperando {wait}s...")
                    logger.warning(f"CHIRPS: 429 en {url}, esperando {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()

                content = _maybe_gunzip(resp.content)
                if not _looks_like_tiff(content):
                    print(f"        200 pero contenido no-TIFF ({content[:20]!r})")
                    break
                print(f"        TIFF válido tras descompresión ({len(content)} bytes)")
                return content
            except requests.exceptions.RequestException as e:
                print(f"         intento {attempt} falló: {e}")
                logger.warning(f"CHIRPS intento {attempt}/{MAX_RETRIES} falló para {url}: {e}")
                time.sleep(2 * attempt)

    print(f"      CHIRPS no disponible para {d.isoformat()}")
    logger.warning(f"CHIRPS: no disponible para {d.isoformat()}")
    return None


def _read_window_from_bytes(content: bytes) -> tuple[np.ndarray, rasterio.Affine] | None:
    minx, miny, maxx, maxy = bbox_window()
    try:
        with MemoryFile(content) as memfile:
            with memfile.open() as src:
                window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                data = src.read(1, window=window)
                window_transform = src.window_transform(window)
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                print(f"        ventana leída: shape={data.shape}, "
                      f"min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")
                return data, window_transform
    except Exception as e:
        print(f"        error leyendo raster en memoria: {e}")
        logger.error(f"CHIRPS: no se pudo leer el raster en memoria: {e}")
        return None


def _read_window_via_vsicurl(d: date) -> tuple[np.ndarray, rasterio.Affine] | None:
    url = _daily_url(d, gz=True)
    vsi_path = f"/vsigzip//vsicurl/{url}"
    minx, miny, maxx, maxy = bbox_window()
    try:
        print(f"      · probando vsicurl: {vsi_path}")
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(vsi_path) as src:
                window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
                data = src.read(1, window=window)
                window_transform = src.window_transform(window)
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                print(f"        ✓ vsicurl OK: shape={data.shape}, "
                      f"min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}")
                return data, window_transform
    except rasterio.errors.RasterioIOError as e:
        print(f"         vsicurl falló: {e}")
        return None


def _read_window_for_date(d: date, method: str = "auto") -> tuple[np.ndarray, rasterio.Affine] | None:
    if method in ("vsicurl", "auto"):
        result = _read_window_via_vsicurl(d)
        if result is not None:
            return result
        if method == "vsicurl":
            print(f"       vsicurl-only falló para {d.isoformat()}")
            logger.warning(f"CHIRPS: vsicurl falló para {d.isoformat()}")
            return None
        print(f"      fallback a download para {d.isoformat()}...")
        logger.info(f"CHIRPS: vsicurl falló para {d.isoformat()}, probando fallback download...")

    content = _download_bytes(d)
    if content is None:
        return None
    return _read_window_from_bytes(content)


def _extract_municipio_values(data: np.ndarray, transform: rasterio.Affine) -> dict[str, float]:
    values = {}
    inv_transform = ~transform
    for nombre, lat, lon in municipios_iter():
        col, row = inv_transform * (lon, lat)
        row, col = int(row), int(col)
        if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
            values[nombre] = float(data[row, col])
        else:
            values[nombre] = np.nan
    return values


def ingest_chirps(
    start: date | None = None,
    end: date | None = None,
    method: str = "auto",
) -> None:
    start = start or DEFAULT_START
    end = end or DEFAULT_END

    total_days = (end - start).days + 1
    print("\n" + "═" * 70)
    print(f"INGESTA CHIRPS")
    print(f"  Rango        : {start.isoformat()}  →  {end.isoformat()}")
    print(f"  Días totales : {total_days}")
    print(f"  Método       : {method}")
    print(f"  Salida       : {PROCESSED_DIR / 'chirps.parquet'}")
    print("═" * 70 + "\n")

    global_ok, global_missing = 0, 0
    dias_procesados = 0

    current_year = start.year
    while current_year <= end.year:
        year_start = max(date(current_year, 1, 1), start)
        year_end = min(date(current_year, 12, 31), end)

        print(f"\n┌── AÑO {current_year} ── {year_start.isoformat()} → {year_end.isoformat()} ──┐")

        rows = []
        n_ok, n_missing = 0, 0
        d = year_start
        year_total = (year_end - year_start).days + 1

        while d <= year_end:
            dias_procesados += 1
            progreso = _progress_bar(dias_procesados, total_days, prefix=f"[{current_year}]")
            print(f"\n{progreso}  → {d.isoformat()}")

            result = _read_window_for_date(d, method=method)
            if result is not None:
                data, transform = result
                municipio_values = _extract_municipio_values(data, transform)
                n_ok += 1
                for municipio, val in municipio_values.items():
                    rows.append({"date": d, "municipio": municipio, "precipitation_mm": val})
                print(f"        ✓ {len(municipio_values)} municipios extraídos")
            else:
                n_missing += 1
                print(f"        ✗ sin datos para {d.isoformat()}")

            d += timedelta(days=1)
            time.sleep(0.1)

        print(f"\n└── AÑO {current_year}: {n_ok}/{year_total} días OK, {n_missing} faltantes ──┘")
        logger.info(f"CHIRPS {current_year}: {n_ok} días OK, {n_missing} días sin datos/fallidos")

        if n_missing > 0 and n_ok == 0:
            print(f"   TODOS los días de {current_year} fallaron. Corre --diagnose.")
            logger.error(
                f"CHIRPS {current_year}: TODOS los días fallaron. Bloqueo sistemático o URL mal "
                f"construida. Corre 'python chirps.py --diagnose'."
            )

        if rows:
            df = pd.DataFrame(rows)
            df["source"] = "chirps"
            df["year"] = current_year

            out_path = PROCESSED_DIR / "chirps.parquet"
            print(f"\n   Guardando {len(df)} filas en {out_path} (particiones source/year)...")
            save_parquet(df, out_path, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
            print(f"   Guardado OK ({len(df)} filas)")
            logger.info(f"CHIRPS {current_year}: {len(df)} filas guardadas")
        else:
            print(f"\n  Año {current_year} sin filas, no se guarda Parquet.")

        global_ok += n_ok
        global_missing += n_missing
        current_year += 1

    print("\n" + "═" * 70)
    print("RESUMEN GLOBAL CHIRPS")
    print(f"  Días OK      : {global_ok}/{total_days}")
    print(f"  Días faltantes: {global_missing}")
    print(f"  Rango        : {start.isoformat()} → {end.isoformat()}")
    print("═" * 70 + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description="Ingesta / diagnóstico de CHIRPS")
    parser.add_argument("--diagnose", action="store_true",
                        help="Corre el diagnóstico detallado sobre una sola fecha y sale.")
    parser.add_argument("--diagnose-date", type=str, default="2023-06-15",
                        help="Fecha (YYYY-MM-DD) a usar en --diagnose. Default: 2023-06-15.")
    parser.add_argument("--start", type=str, default=None,
                        help=f"YYYY-MM-DD. Default: {DEFAULT_START.isoformat()} (2020-01-01).")
    parser.add_argument("--end", type=str, default=None,
                        help=f"YYYY-MM-DD. Default: {DEFAULT_END.isoformat()} (ayer).")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "vsicurl", "download"],
                        help="Método de lectura. 'auto' intenta vsicurl y cae a download si falla.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.diagnose:
        diag_date = date.fromisoformat(args.diagnose_date)
        diagnose_date(diag_date)
    else:
        ingest_chirps(
            start=date.fromisoformat(args.start) if args.start else None,
            end=date.fromisoformat(args.end) if args.end else None,
            method=args.method,
        )