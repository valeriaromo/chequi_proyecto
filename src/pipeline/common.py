import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from config import settings 

OUT = ROOT / "outputs_fase2" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

RAIN_COL = "precipitation_mm" 
RAIN_THRESH = 1.0            # evento = lluvia >= 1 mm
TMIN_COL = "T2M_MIN"
HORIZON = 1
FOLDS = [2023, 2024, 2025, 2026]


def load_master():
    d = Path(settings.MASTER_DIR)
    df = pd.read_parquet(d / "master_table.parquet")
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df:
        df["year"] = df["date"].dt.year
    return df.sort_values(["municipio", "date"]).reset_index(drop=True)


def rain_target(df, h=HORIZON):
    fut = df.groupby("municipio", observed=True)[RAIN_COL].shift(-h)
    return (fut >= RAIN_THRESH).astype(float).where(fut.notna())


def tmin_target(df, h=HORIZON):
    return df.groupby("municipio", observed=True)[TMIN_COL].shift(-h)


def feature_cols(df):
    drop = {"date", "municipio", "year"}
    cols = [c for c in df.columns
            if c not in drop and "aod" not in c.lower() and not c.startswith(("target_", "y_"))]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def train_mask(df, Y, h=HORIZON):
# el target de la última fila (fecha + h) debe caer antes del 1-ene del año Y
    cutoff = pd.Timestamp(year=Y, month=1, day=1)
    return (df["date"] + pd.Timedelta(days=h)) < cutoff