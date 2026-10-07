from pathlib import Path
from datetime import date

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"               
PROCESSED_DIR = DATA_DIR / "processed"    
MASTER_DIR = DATA_DIR / "master"          

for d in (RAW_DIR, PROCESSED_DIR, MASTER_DIR):
    d.mkdir(parents=True, exist_ok=True)

BBOX_AGS = {
    "lon_min": -102.85,
    "lon_max": -101.85,
    "lat_min": 21.55,
    "lat_max": 22.45,
}


MUNICIPIOS_AGS = {
    "Aguascalientes":        {"lat": 21.8853, "lon": -102.2916},
    "Asientos":               {"lat": 22.2333, "lon": -102.0833},
    "Calvillo":               {"lat": 21.8461, "lon": -102.7194},
    "Cosío":                  {"lat": 22.3667, "lon": -102.3000},
    "Jesús María":            {"lat": 21.9611, "lon": -102.3444},
    "Pabellón de Arteaga":    {"lat": 22.1500, "lon": -102.2667},
    "Rincón de Romos":        {"lat": 22.2333, "lon": -102.3167},
    "San José de Gracia":     {"lat": 22.1667, "lon": -102.5833},
    "Tepezalá":               {"lat": 22.2333, "lon": -102.1667},
    "El Llano":               {"lat": 21.9167, "lon": -102.1500},
    "San Francisco de los Romo": {"lat": 22.0833, "lon": -102.2667},
}

FECHA_INICIO = date(2020, 1, 1)
FECHA_FIN = date.today()

NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

NASA_POWER_PARAMS = [
    "T2M_MAX",   # temperatura máxima
    "T2M_MIN",   # temperatura mínima
    "T2M",       # temperatura promedio
    "RH2M",      # humedad relativa
    "WS2M",      # velocidad del viento
    "ALLSKY_SFC_SW_DWN",  # radiación solar
    "PS",        # presión atmosférica
    "PRECTOTCORR",  # precipitación (corregida)
]

CHIRPS_DAILY_COG_TEMPLATE = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/"
    "{year}/chirps-v2.0.{year}.{month:02d}.{day:02d}.tif.gz"
)


OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPENAQ_PARAMETRO_PRINCIPAL = "pm25"


import os
INEGI_TOKEN = os.environ.get("INEGI_TOKEN")  # opcional para descargar datos de INEGI vía API sino se usan archivos locales.
INEGI_ENTIDAD_AGS = "01"  # clave AGEE de Aguascalientes

INEGI_MUNICIPIOS_AGS = {
    "Aguascalientes":             "001",
    "Asientos":                   "002",
    "Calvillo":                   "003",
    "Cosío":                      "004",
    "Jesús María":                "005",
    "Pabellón de Arteaga":        "006",
    "Rincón de Romos":            "007",
    "San José de Gracia":         "008",
    "Tepezalá":                   "009",
    "El Llano":                   "010",
    "San Francisco de los Romo":  "011",
}

OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RIO_SEARCH_RADIUS_M = 20000
CENAPRED_ARCGIS_MAPSERVER = (
    "https://sigagis.conagua.gob.mx/ArcGIS/rest/services/Caracterizaci%C3%B3n/MapServer"
)
CENAPRED_ESTADO_FILTRO = "Aguascalientes"

GEE_PROJECT = "chequiproyecto"

from datetime import date, timedelta
FECHA_FIN_MASTER = date(2026, 7, 31) 
FECHA_INICIO_MASTER = date(2020, 1, 1)


PARQUET_COMPRESSION = "zstd"   # mejor ratio que snappy para series de tiempo
PARTITION_COLS = ["source", "year"]

MASTER_TABLE_PATH = MASTER_DIR/ "master_table.parquet"
REPORTS_DIR = MASTER_DIR/ "reports"
TARGETS = ["T2M_MAX", "T2M_MIN", "precipitation_mm"]
HORIZONTES = [1, 3, 7]
MIN_TRAIN_YEARS = 3   # años mínimos de entrenamiento antes del primer fold
MARGEN_AOD = 0.01     # AOD se adopta solo si baja el MAE >1 % y gana en >50 % de folds
SEED = 42
