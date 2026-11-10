from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent
ROOT = MODELS_DIR.parents[2]
TABLES = {"master": ROOT / "data" / "master" / "master_table.parquet","pca": ROOT / "data" / "master" / "master_table_pca.parquet",}
MASTER_DIR = ROOT / "data" / "master"
FEATURE_SETS = ("base", "aod", "static", "all")
STATIC_COLUMNS = ["geom_area_km2", "pob", "viv"]
CYCLONE_COLUMNS = [
    "l25_declaratorias_de_desastres_por_ciclones__n",
    "l25_declaratorias_de_desastres_por_ciclones__area_max",
    "l25_declaratorias_de_desastres_por_ciclones__edo_num_max",
    "l25_declaratorias_de_desastres_por_ciclones__mpo_num_max",
    "l25_declaratorias_de_desastres_por_ciclones__ciclones_f_max",
]
EXTRA_BY_SET = {"base": [], "aod": [], "static": STATIC_COLUMNS, "all": STATIC_COLUMNS + CYCLONE_COLUMNS}

def output_dir(table, feature_set="base"):
    parts = [p for p in (table if table != "master" else "", feature_set if feature_set != "base" else "") if p]
    path = MODELS_DIR.joinpath("outputs", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path

HORIZON = 1        # days ahead (h=1, the pilot scope)
FOLD_YEARS = (2023, 2024, 2025, 2026)  # walk forward test years, expanding train window
RANDOM_STATE = 42

RAIN_COL = "precipitation_mm"
RAIN_THRESHOLD_MM = 1.0
TEMP_COL = "T2M_MIN"

# ns si si hay year
KEY_COLS = ("date", "municipio", "year")
EXCLUDED_SUBSTRINGS = ("aod", "target") # nos cae mal AOD, y target es el futuro que queremos predecir
