from __future__ import annotations
import argparse
import gzip
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from rasterio.windows import Window, from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (CHIRPS_DAILY_COG_TEMPLATE,FECHA_FIN,FECHA_INICIO,MUNICIPIOS_AGS,PARQUET_COMPRESSION,PROCESSED_DIR,)
from src.utils.geo_utils import bbox_window, load_municipios_geom  
from src.utils.io_utils import save_parquet  
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("chirps") 

SOURCE = "chirps"
OUT_PATH = PROCESSED_DIR / "chirps.parquet"
GEOM_PATH = PROCESSED_DIR / "inegi.parquet" / "geometria_municipios.parquet"


def _to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])

DEFAULT_START = _to_date(FECHA_INICIO)
DEFAULT_END = min(_to_date(FECHA_FIN), date.today() - timedelta(days=1)) # recorta la fecha a ayer porque chirps tiene latencia de 1 día

HTTP_HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ""(KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"),"Accept": "*/*",}
GDAL_ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR","CPL_VSIL_CURL_USE_HEAD": "NO","VSI_CACHE": "TRUE",
    "GDAL_HTTP_USERAGENT": HTTP_HEADERS["User-Agent"],"GDAL_HTTP_TIMEOUT": "30","GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1","GDAL_INGESTED_BYTES_AT_OPEN": "65536",  # para que el cache de vsicurl no se llene con el .gz completo
    "CPL_VSIL_CURL_CACHE_SIZE": "200000000","GDAL_HTTP_MULTIPLEX": "YES","GDAL_HTTP_VERSION": "2",
}
MAX_RETRIES = 3
REQUEST_TIMEOUT_S = 30
GZIP_MAGIC = b"\x1f\x8b"
TIFF_MAGICS = (b"II*\x00", b"MM\x00*")

RasterWindow = tuple[np.ndarray, "rasterio.Affine", "rasterio.crs.CRS"]

def _daily_url(d: date, gz: bool = True) -> str:
    url = CHIRPS_DAILY_COG_TEMPLATE.format(year=d.year, month=d.month, day=d.day)
    if gz:
        return url
    return url[:-3] if url.endswith(".gz") else url

def _url_variants(d: date) -> list[str]:
    urls = [_daily_url(d, gz=True), _daily_url(d, gz=False)]
    return list(dict.fromkeys(urls))

def _vsi_path(url: str) -> str:
    return f"/vsigzip//vsicurl/{url}" if url.endswith(".gz") else f"/vsicurl/{url}"

def _looks_like_tiff(content: bytes) -> bool:
    return content[:4] in TIFF_MAGICS

def _maybe_gunzip(content: bytes) -> bytes:
    return gzip.decompress(content) if content[:2] == GZIP_MAGIC else content

def _read_window(src: rasterio.DatasetReader) -> RasterWindow | None:
    minx, miny, maxx, maxy = bbox_window()
    w = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    col0, row0 = int(np.floor(w.col_off)), int(np.floor(w.row_off))
    col1, row1 = int(np.ceil(w.col_off + w.width)), int(np.ceil(w.row_off + w.height))
    win = Window(col0, row0, col1 - col0, row1 - row0)
    data = src.read(1, window=win).astype("float32")
    if src.nodata is not None:
        data[data == src.nodata] = np.nan
    data[data < 0] = np.nan  # elimina sentinelas negativos (CHIRPS usa -9999 y -1 para nodata)
    if np.isnan(data).all():
        logger.warning("ventana completamente nodata")
        return None
    return data, src.window_transform(win), src.crs


def _read_remote(d: date) -> RasterWindow | None:
    for url in _url_variants(d):
        path = _vsi_path(url)
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(path) as src:
                result = _read_window(src)
                if result is not None:
                    return result
        except Exception as e:  
            logger.debug("vsicurl falló (%s): %s", path, e)
    return None


def _download_bytes(d: date, session: requests.Session) -> bytes | None:
    for url in _url_variants(d):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = session.get(url, timeout=REQUEST_TIMEOUT_S)
                if resp.status_code == 404:
                    break  # prueba la otra variante de URL
                if resp.status_code in (401, 403):
                    logger.error("CHIRPS %s en %s: bloqueo, no se reintenta", resp.status_code, url)
                    break
                if resp.status_code == 429:
                    wait = 5 * attempt
                    logger.warning("CHIRPS 429 en %s, esperando %ss", url, wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                content = _maybe_gunzip(resp.content)
                if not _looks_like_tiff(content):
                    logger.warning("CHIRPS %s: 200 pero no es TIFF (%r)", url, content[:20])
                    break
                return content
            except (requests.RequestException, OSError, EOFError) as e:
                logger.warning("CHIRPS intento %d/%d falló (%s): %s", attempt, MAX_RETRIES, url, e)
                time.sleep(2 * attempt)
    return None

def _read_from_bytes(content: bytes) -> RasterWindow | None:
    try:
        with MemoryFile(content) as mem, mem.open() as src:
            return _read_window(src)
    except Exception as e:
        logger.error("CHIRPS: no se pudo leer el raster en memoria: %s", e)
        return None

def _read_window_for_date(d: date, method: str, session: requests.Session) -> RasterWindow | None:
    if method in ("auto", "vsicurl"):
        result = _read_remote(d)
        if result is not None or method == "vsicurl":
            return result
        logger.info("%s: vsicurl falló, fallback a download", d.isoformat())
    content = _download_bytes(d, session)
    return None if content is None else _read_from_bytes(content)

_MASK_CACHE: dict[tuple, dict[str, np.ndarray]] = {}

def _get_masks(shape: tuple[int, int], transform, crs) -> dict[str, np.ndarray]:
    key = (shape, tuple(transform)[:6], str(crs))
    if key in _MASK_CACHE:
        return _MASK_CACHE[key]

    gdf = load_municipios_geom()
    if gdf.crs is None:
        logger.warning("geometría sin CRS; se asume el del raster (%s). Verifica inegi.py", crs)
    elif crs is not None and gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    masks: dict[str, np.ndarray] = {}
    for _, row in gdf.iterrows():
        geom = row["geometry"]
        m = geometry_mask([geom], out_shape=shape, transform=transform, all_touched=True, invert=True)
        if not m.any():
            pt = geom.representative_point()
            # para municipios muy pequeños que no intersectan la ventana, se toma el punto representativo y se marca el pixel más cercano
            r, c = rowcol(transform, pt.x, pt.y)
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                m[r, c] = True
            else:
                logger.warning("%s no intersecta la ventana", row["municipio"])
        masks[row["municipio"]] = m
    _MASK_CACHE.clear()
    _MASK_CACHE[key] = masks
    return masks
_MASK_LOCK = threading.Lock()

def _get_masks_safe(shape, transform, crs) -> dict[str, np.ndarray]:
    with _MASK_LOCK:
        return _get_masks(shape, transform, crs)

def _extract_municipio_values(data: np.ndarray, transform, crs) -> dict[str, float]:
    masks = _get_masks_safe(data.shape, transform, crs)
    out = {}
    for mun, m in masks.items():
        vals = data[m]
        vals = vals[~np.isnan(vals)]
        out[mun] = float(vals.mean()) if vals.size else np.nan
    return out

def _check_prereqs() -> None:
    if not GEOM_PATH.exists():
        raise FileNotFoundError(
            f"Falta {GEOM_PATH}. Corre inegi.py primero (orden de ejecución obligatorio)."
        )
    geom_names = set(load_municipios_geom()["municipio"])
    catalog = set(MUNICIPIOS_AGS)
    if geom_names != catalog:
        raise ValueError(
            "Nombres de municipio en la geometría no coinciden con MUNICIPIOS_AGS. "
            f"Solo en geometría: {sorted(geom_names - catalog)} | "
            f"solo en catálogo: {sorted(catalog - geom_names)}"
        )

def _save_year(df: pd.DataFrame, year: int) -> None:
    if OUT_PATH.exists():
        old = pd.read_parquet(OUT_PATH, filters=[("year", "==", year)])
        if not old.empty:
            old = old.drop(columns=[c for c in ("source", "year") if c in old.columns])
            old["date"] = pd.to_datetime(old["date"])
            df = pd.concat([old, df.drop(columns=["source", "year"])], ignore_index=True)
            df = df.drop_duplicates(["date", "municipio"], keep="last")
            df["source"] = SOURCE
            df["year"] = year
    df = df.sort_values(["date", "municipio"]).reset_index(drop=True)
    save_parquet(df, OUT_PATH, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    logger.info("CHIRPS %d: %d filas guardadas", year, len(df))

_tls = threading.local()


def _session() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
        _tls.s.headers.update(HTTP_HEADERS)
    return _tls.s

def _existing_dates(year: int) -> set[date]:
    if not OUT_PATH.exists():
        return set()
    try:
        old = pd.read_parquet(OUT_PATH, columns=["date"], filters=[("year", "==", year)])
    except Exception as e:
        logger.warning("no se pudieron leer fechas existentes de %d: %s", year, e)
        return set()
    return set(pd.to_datetime(old["date"]).dt.date)

def _fetch_day(d: date, method: str) -> tuple[date, dict[str, float] | None]:
    try:
        result = _read_window_for_date(d, method, _session())
        return d, None if result is None else _extract_municipio_values(*result)
    except Exception as e:
        logger.error("CHIRPS %s: error inesperado: %s", d, e)
        return d, None

def ingest_chirps(start: date | None = None,end: date | None = None,method: str = "auto",workers: int = 8,force: bool = False,) -> None:
    start, end = start or DEFAULT_START, end or DEFAULT_END
    if start > end:
        raise ValueError(f"start {start} > end {end}")
    _check_prereqs()

    logger.info("INGESTA CHIRPS %s → %s (método=%s, workers=%d, force=%s) → %s",
                start, end, method, workers, force, OUT_PATH)
    g_ok = g_missing = g_skipped = 0

    for year in range(start.year, end.year + 1):
        y0, y1 = max(date(year, 1, 1), start), min(date(year, 12, 31), end)
        days = [y0 + timedelta(n) for n in range((y1 - y0).days + 1)]
        have = set() if force else _existing_dates(year)
        todo = [d for d in days if d not in have]
        g_skipped += len(days) - len(todo)
        if not todo:
            logger.info("CHIRPS %d: completo, se omite", year)
            continue

        logger.info("CHIRPS %d: %d días por descargar (%d ya existentes)", year, len(todo), len(days) - len(todo))
        rows: list[dict] = []
        y_ok = y_missing = 0
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [ex.submit(_fetch_day, d, method) for d in todo]
            for i, fut in enumerate(as_completed(futures), 1):
                d, values = fut.result()
                if values is not None:
                    rows.extend({"date": d, "municipio": m, "precipitation_mm": v} for m, v in values.items())
                    y_ok += 1
                else:
                    y_missing += 1
                    logger.warning("CHIRPS %s: sin datos", d)
                if i % 50 == 0 or i == len(todo):
                    logger.info("[%d] %d/%d (%.1f%%)", year, i, len(todo), 100 * i / len(todo))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            if rows:  # guarda lo avanzado aunque haya Ctrl+C o excepción
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"])
                df["source"] = SOURCE
                df["year"] = year
                _save_year(df, year)

        logger.info("CHIRPS %d: %d OK, %d faltantes", year, y_ok, y_missing)
        if y_ok == 0:
            logger.error("CHIRPS %d: todos los días fallaron. Corre --diagnose.", year)
        g_ok += y_ok
        g_missing += y_missing

    logger.info("RESUMEN CHIRPS: %d descargados, %d faltantes, %d ya existentes", g_ok, g_missing, g_skipped)

def diagnose_date(d: date) -> None:
    url = _daily_url(d, gz=True)
    print(f"DIAGNÓSTICO CHIRPS {d} | URL: {url}")

    print("[1/3] HEAD")
    try:
        r = requests.head(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_S, allow_redirects=True)
        print(f"  status={r.status_code} type={r.headers.get('Content-Type')} len={r.headers.get('Content-Length')}")
    except requests.RequestException as e:
        print(f"  ERROR: {e}")

    print("[2/3] download + gunzip + rasterio")
    session = requests.Session()
    session.headers.update(HTTP_HEADERS)
    content = _download_bytes(d, session)
    if content is None:
        print("  descarga fallida")
    else:
        res = _read_from_bytes(content)
        print("  OK" if res else "  lectura fallida", f"shape={res[0].shape}" if res else "")

    print("[3/3] vsicurl")
    res = _read_remote(d)
    print("  OK" if res else "  falló", f"shape={res[0].shape}" if res else "")


def _parse_args():
    p = argparse.ArgumentParser(description="Ingesta / diagnóstico de CHIRPS")
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--diagnose-date", default="2023-06-15")
    p.add_argument("--start", default=None, help=f"YYYY-MM-DD. Default: {DEFAULT_START}")
    p.add_argument("--end", default=None, help=f"YYYY-MM-DD. Default: {DEFAULT_END}")
    p.add_argument("--method", default="auto", choices=["auto", "vsicurl", "download"])
    p.add_argument("--workers", type=int, default=8, help="Hilos concurrentes (I/O bound).")
    p.add_argument("--force", action="store_true", help="Re-descarga aunque la fecha ya exista.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.diagnose:
        diagnose_date(date.fromisoformat(args.diagnose_date))
    else:
        ingest_chirps(
            start=date.fromisoformat(args.start) if args.start else None,
            end=date.fromisoformat(args.end) if args.end else None,
            method=args.method,
            workers=args.workers,
            force=args.force,
        )