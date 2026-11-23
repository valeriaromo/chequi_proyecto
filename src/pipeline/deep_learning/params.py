import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from config.settings import MASTER_DIR  

OUT_DIR = Path(__file__).parent / "outputs"
FIG_DIR = OUT_DIR / "figures"
MASTER_FILE = MASTER_DIR / "master_table.parquet"     
FASE2_DIR = ROOT / "outputs_fase2"

SEED, WINDOW, HORIZON = 42, 30, 1
TEST_YEARS = [2023, 2024, 2025, 2026]                  # expanding walk forward val = year before test
HIDDEN, LSTM_LAYERS, DROPOUT, DENSE = 64, 1, 0.3, 32
EPOCHS, BATCH, LR, PATIENCE = 50, 256, 1e-3, 5
MIN_GAIN = 0.02                                        # min mean gain on primary metric to recommend DL

EXCLUDE_COLS = {"date", "municipio", "year", "qc_aod_imputed"}
EXCLUDE_PREFIXES = ("aod", "y_")                       # AOD frozen out (usar_aod=False) y_ = targets

# add an entry and append its key to ACTIVE_TARGETS.
# kind: clf (binary via thr)
# reg. weight: loss weight. baseline_*: ML results to compare against.
TARGETS = {
    "rain_occ": dict(kind="clf", col="precipitation_mm", thr=1.0, weight=1.0,baseline_csv=FASE2_DIR / "rain_eval.csv", baseline_filter={}),
    "t2m_min": dict(kind="reg", col="T2M_MIN", thr=None, weight=1.0,baseline_csv=FASE2_DIR / "scope_T2M_MIN.csv", baseline_filter={}),
}
ACTIVE_TARGETS = ["rain_occ", "t2m_min"]  

ML_TRAIN_SECONDS = None                                # fill in XGBoost train time if known