from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MASTER = ROOT / "data" / "master" / "master_table.parquet"  
OUT = HERE / "outputs"
SEED = 42

# Raw daily variables only: no AOD, lags, rolling means, static, cyclone or target columns
VARS = ["T2M", "T2M_MAX", "T2M_MIN","precipitation_mm", "PRECTOTCORR", "rain_3d", "rain_7d","rain_30d", "rain_90d", "dias_sin_lluvia","RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN"]
SUM_VARS = ["precipitation_mm"]  # monthly sum; every other variable uses the monthly mean
MIN_DAYS = 20                 # drop partial months (e.g. 2026)
PCA_VAR = 0.90                # cumulative variance retained
K_RANGE = range(2, 11)
K_FINAL_MIN = 3               # k=2 is usually just a seasonal split
BOOT_RUNS = 30

# readiness thresholds
READY_SIL, READY_ARI, READY_NEG = 0.40, 0.80, 0.10
COND_SIL, COND_ARI = 0.25, 0.60