from pathlib import Path
MODELS_DIR = Path(__file__).resolve().parent
ROOT = MODELS_DIR.parents[2]
TABLES = {"master": ROOT / "data" / "master" / "master_table.parquet","pca": ROOT / "data" / "master" / "master_table_pca.parquet",}

HORIZON = 1        # days ahead (h=1, the pilot scope)
FOLD_YEARS = (2023, 2024, 2025, 2026)  # walk forward test years, expanding train window
RANDOM_STATE = 42

RAIN_COL = "precipitation_mm"
RAIN_THRESHOLD_MM = 1.0
TEMP_COL = "T2M_MIN"

# ns si si hay year
KEY_COLS = ("date", "municipio", "year")
EXCLUDED_SUBSTRINGS = ("aod", "target") # nos cae mal AOD, y target es el futuro que queremos predecir


def output_dir(table: str) -> Path:
    path = MODELS_DIR / "outputs" / ("" if table == "master" else table)
    path.mkdir(parents=True, exist_ok=True)
    return path
