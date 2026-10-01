"""
Esta ingesta guarda valores CRUDOS (NaN donde no hay retrieval válido). La imputación NO se hace
aquí: va en una etapa posterior, ajustada solo con datos de train. Las columnas qc_aod_imputed y
aod_imputed_method se escriben como False / "none" para fijar el esquema.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd
try:
    import ee
except ImportError as e:
    raise SystemExit("Falta earthengine-api: pip install earthengine-api") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import settings 
from config.settings import (FECHA_FIN,FECHA_INICIO,MUNICIPIOS_AGS,PARQUET_COMPRESSION,PROCESSED_DIR,)
from src.utils.geo_utils import bbox_window, load_municipios_geom 
from src.utils.io_utils import save_parquet 

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("maiac_aod")

SOURCE = "maiac_aod"
OUT_PATH = PROCESSED_DIR / "maiac_aod.parquet"
GEOM_PATH = PROCESSED_DIR / "inegi.parquet" / "geometria_municipios.parquet"

COLLECTION = getattr(settings, "MAIAC_GEE_COLLECTION", "MODIS/061/MCD19A2_GRANULES")
GEE_PROJECT = getattr(settings, "GEE_PROJECT", None) or os.environ.get("GEE_PROJECT")

LATENCY_DAYS = 7 
CHUNK_DAYS = 10      
QUOTA_RETRIES = 6     
AOD_SCALE = 0.001
AOD_MAX = 5.0         
SIMPLIFY_DEG = 0.0005  

def _to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return date.fromisoformat(str(v)[:10])

DEFAULT_START = _to_date(FECHA_INICIO)
DEFAULT_END = min(_to_date(FECHA_FIN), date.today() - timedelta(days=LATENCY_DAYS))

def _ee_init() -> None:
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception as e:
        raise RuntimeError(
            "No se pudo inicializar Earth Engine. Corre `earthengine authenticate` y define "
            "GEE_PROJECT en config/settings.py o como variable de entorno."
        ) from e

def _municipios_fc() -> "ee.FeatureCollection":
    gdf = load_municipios_geom()[["municipio", "geometry"]].copy()
    if gdf.crs is None:
        logger.warning("geometría sin CRS; se asume EPSG:4326. Verifica inegi.py")
    else:
        gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf.geometry.simplify(SIMPLIFY_DEG, preserve_topology=True)
    return ee.FeatureCollection(json.loads(gdf.to_json()))

def _process_granule(img: "ee.Image") -> "ee.Image":
    qa = img.select("AOD_QA")
    cloud_clear = qa.bitwiseAnd(0b111).eq(1)
    land_water = qa.rightShift(3).bitwiseAnd(0b11).lte(1)
    qa_aod = qa.rightShift(8).bitwiseAnd(0b1111)
    aod = img.select("Optical_Depth_055").multiply(AOD_SCALE)
    ok = cloud_clear.And(land_water).And(qa_aod.lte(1)).And(aod.gte(0)).And(aod.lte(AOD_MAX))
    return ee.Image.cat([aod.rename("aod"), qa_aod.eq(0).rename("best")]).updateMask(ok)


def _daily_image(day: "ee.Date", region: "ee.Geometry") -> "ee.Image":
    coll = (
        ee.ImageCollection(COLLECTION)
        .filterDate(day, day.advance(1, "day"))
        .filterBounds(region)
        .map(_process_granule)
    )
    blank = ee.Image.constant([0, 0]).rename(["aod", "best"]).updateMask(0)
    merged = coll.merge(ee.ImageCollection([blank]))
    return merged.select("aod").mean().addBands(merged.select("best").max())


def _native_projection(region: "ee.Geometry") -> dict:
    img = (
        ee.ImageCollection(COLLECTION)
        .filterDate("2023-01-01", "2023-01-10")
        .filterBounds(region)
        .first()
    )
    return img.select("AOD_QA").projection().getInfo()


def _fetch_chunk(days: list[date], fc, region, proj: dict) -> list[dict]:
    reducer = ee.Reducer.mean().unweighted().combine(ee.Reducer.count().unweighted(), sharedInputs=True)
    def per_day(s):
        s = ee.String(s)
        img = _daily_image(ee.Date(s), region)
        res = img.reduceRegions(
            collection=fc, reducer=reducer, crs=proj["crs"], crsTransform=proj["transform"]
        )
        return res.map(lambda f: ee.Feature(f).setGeometry(None).set("date", s))
    dates = ee.List([d.isoformat() for d in days])
    out = ee.FeatureCollection(dates.map(per_day)).flatten()
    return [f["properties"] for f in out.getInfo()["features"]]


def _is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(k in msg for k in ("concurrent", "429", "quota", "rate limit", "too many requests"))


def _fetch_with_retry(days: list[date], fc, region, proj: dict) -> list[dict]:
    for attempt in range(1, QUOTA_RETRIES + 1):
        try:
            return _fetch_chunk(days, fc, region, proj)
        except Exception as e:
            if _is_quota_error(e):
                wait = min(120, 10 * 2 ** (attempt - 1)) + random.uniform(0, 5)
                logger.warning("GEE %s→%s: límite de concurrencia, espera %.0fs (%d/%d)",
                               days[0], days[-1], wait, attempt, QUOTA_RETRIES)
                time.sleep(wait)
                continue
            logger.warning("GEE %s→%s (%d días) falló: %s", days[0], days[-1], len(days), e)
            break
    else:
        logger.error("GEE %s→%s: agotó reintentos por cuota; se reintentará en la próxima corrida",
                     days[0], days[-1])
        return []

    if len(days) > 1:
        mid = len(days) // 2
        return (_fetch_with_retry(days[:mid], fc, region, proj)
                + _fetch_with_retry(days[mid:], fc, region, proj))
    logger.error("GEE %s: falló definitivamente, se reintentará en la próxima corrida", days[0])
    return []

def _to_frame(props: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(props).reindex(columns=["date", "municipio", "aod_mean", "aod_count", "best_mean"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "municipio": df["municipio"].astype(str),
        "aod_055_mean": df["aod_mean"].astype("float32"),
        "aod_055_count": df["aod_count"].fillna(0).astype("int32"),
        "aod_qa_best_pct": (df["best_mean"] * 100).astype("float32"),  # % de píxeles válidos con QA=0
    })
    out["source"] = SOURCE
    out["year"] = out["date"].dt.year
    out["qc_aod_imputed"] = False
    out["aod_imputed_method"] = "none"
    return out


def _check_prereqs() -> None:
    if not GEOM_PATH.exists():
        raise FileNotFoundError(f"Falta {GEOM_PATH}. Corre inegi.py primero (orden obligatorio).")
    geom_names = set(load_municipios_geom()["municipio"])
    catalog = set(MUNICIPIOS_AGS)
    if geom_names != catalog:
        raise ValueError(
            "Nombres de municipio en la geometría no coinciden con MUNICIPIOS_AGS. "
            f"Solo en geometría: {sorted(geom_names - catalog)} | solo en catálogo: {sorted(catalog - geom_names)}"
        )


def _write_year_partition(df: pd.DataFrame, year: int) -> None:
    tmp_root = OUT_PATH.with_name(f"{OUT_PATH.name}.tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    save_parquet(df, tmp_root, partition_cols=["source", "year"], compression=PARQUET_COMPRESSION)
    new_part = tmp_root / f"source={SOURCE}" / f"year={year}"
    final_part = OUT_PATH / f"source={SOURCE}" / f"year={year}"
    bak = OUT_PATH.with_name(f"{OUT_PATH.name}.bak_{year}")
    final_part.parent.mkdir(parents=True, exist_ok=True)
    if bak.exists():
        shutil.rmtree(bak)
    if final_part.exists():
        final_part.rename(bak)
    new_part.rename(final_part)
    shutil.rmtree(bak, ignore_errors=True)
    shutil.rmtree(tmp_root, ignore_errors=True)

def _save_year(df: pd.DataFrame, year: int) -> None:
    if OUT_PATH.exists():
        old = pd.read_parquet(OUT_PATH, filters=[("year", "==", year)])
        if not old.empty:
            old = old.drop(columns=[c for c in ("source", "year") if c in old.columns])
            old["date"] = pd.to_datetime(old["date"])
            df = pd.concat([old, df.drop(columns=["source", "year"])], ignore_index=True)
            df["source"] = SOURCE
            df["year"] = year
    df = df.drop_duplicates(["date", "municipio"], keep="last").sort_values(["date", "municipio"])
    _write_year_partition(df.reset_index(drop=True), year)
    logger.info("MAIAC %d: %d filas guardadas", year, len(df))


def _existing_dates(year: int) -> set[date]:
    if not OUT_PATH.exists():
        return set()
    try:
        old = pd.read_parquet(OUT_PATH, columns=["date"], filters=[("year", "==", year)])
    except Exception as e:
        logger.warning("no se pudieron leer fechas existentes de %d: %s", year, e)
        return set()
    return set(pd.to_datetime(old["date"]).dt.date)


def _context():
    _ee_init()
    fc = _municipios_fc()
    minx, miny, maxx, maxy = bbox_window()
    region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])
    return fc, region, _native_projection(region)

def ingest_maiac(
    start: date | None = None,
    end: date | None = None,
    workers: int = 2,
    chunk_days: int = CHUNK_DAYS,
    force: bool = False,
) -> None:
    start, end = start or DEFAULT_START, end or DEFAULT_END
    if start > end:
        raise ValueError(f"start {start} > end {end}")
    _check_prereqs()
    fc, region, proj = _context()
    logger.info("INGESTA MAIAC %s → %s (workers=%d, chunk=%d días, force=%s) → %s",
                start, end, workers, chunk_days, force, OUT_PATH)

    for year in range(start.year, end.year + 1):
        y0, y1 = max(date(year, 1, 1), start), min(date(year, 12, 31), end)
        days = [y0 + timedelta(n) for n in range((y1 - y0).days + 1)]
        have = set() if force else _existing_dates(year)
        todo = [d for d in days if d not in have]
        if not todo:
            logger.info("MAIAC %d: completo, se omite", year)
            continue

        chunks = [todo[i:i + chunk_days] for i in range(0, len(todo), chunk_days)]
        logger.info("MAIAC %d: %d días en %d bloques (%d ya existentes)",
                    year, len(todo), len(chunks), len(days) - len(todo))

        frames: list[pd.DataFrame] = []
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [ex.submit(_fetch_with_retry, c, fc, region, proj) for c in chunks]
            for i, fut in enumerate(as_completed(futures), 1):
                props = fut.result()
                if props:
                    frames.append(_to_frame(props))
                logger.info("[%d] bloque %d/%d", year, i, len(chunks))
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                unknown = set(df["municipio"]) - set(MUNICIPIOS_AGS)
                if unknown:
                    raise ValueError(f"municipios fuera del catálogo: {sorted(unknown)}")
                got = set(df["date"].dt.date)
                missing = [d for d in todo if d not in got]
                valid_pct = 100 * (df["aod_055_count"] > 0).mean()
                logger.info("MAIAC %d: %d días recibidos, %d sin respuesta de GEE, %.1f%% celdas con AOD válido",
                            year, len(got), len(missing), valid_pct)
                _save_year(df, year)
            else:
                logger.error("MAIAC %d: sin datos. Corre --diagnose.", year)

def diagnose(d: date) -> None:
    fc, region, proj = _context()
    n = (ee.ImageCollection(COLLECTION)
         .filterDate(d.isoformat(), (d + timedelta(days=1)).isoformat())
         .filterBounds(region).size().getInfo())
    print(f"proyección nativa: {proj['crs']} | granules {d}: {n}")
    props = _fetch_chunk([d], fc, region, proj)
    print(_to_frame(props).drop(columns=["source", "year"]).to_string(index=False))


def _parse_args():
    p = argparse.ArgumentParser(description="Ingesta MAIAC MCD19A2 AOD (GEE)")
    p.add_argument("--start", default=None, help=f"YYYY-MM-DD. Default: {DEFAULT_START}")
    p.add_argument("--end", default=None, help=f"YYYY-MM-DD. Default: {DEFAULT_END}")
    p.add_argument("--workers", type=int, default=2, help="Bloques concurrentes hacia GEE.")
    p.add_argument("--chunk-days", type=int, default=CHUNK_DAYS)
    p.add_argument("--force", action="store_true", help="Re-descarga aunque la fecha ya exista.")
    p.add_argument("--diagnose", action="store_true", help="Procesa un solo día y lo imprime.")
    p.add_argument("--diagnose-date", default="2023-03-15")
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    if args.diagnose:
        diagnose(date.fromisoformat(args.diagnose_date))
    else:
        ingest_maiac(
            start=date.fromisoformat(args.start) if args.start else None,
            end=date.fromisoformat(args.end) if args.end else None,
            workers=args.workers,
            chunk_days=args.chunk_days,
            force=args.force,
        )