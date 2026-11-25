from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
MASTER = ROOT / "data" / "master" / "master_table.parquet"
OUT = HERE / "outputs"
IMG, JS, MODELS = OUT / "images", OUT / "json", OUT / "models"

H = 1                 # forecast horizon (days)
SEED = 42
TEST_YEAR = 2025      # train on years < TEST_YEAR (purged by H days), explain TEST_YEAR
THR = 0.5             # classification threshold for case selection
RAIN_MM = 1.0
TOP_K = 15            # features shown in plots / matrices
LIME_K = 10           # features per LIME explanation
LIME_N = 3000         # LIME perturbation samples

TARGETS = {
    "rain_occ": dict(kind="clf", src="precipitation_mm",      # EDIT: CHIRPS daily precip column
                     unit="log-odds of rain (>=1 mm) tomorrow"),
    "t2m_min": dict(kind="reg", src="T2M_MIN",
                    unit="tomorrow's T2M_MIN (°C)"),
}

# base hyper-parameters; max_depth, min_child_weight, learning_rate, n_estimators are tuned in train.py
PARAMS = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, random_state=SEED, n_jobs=-1)

# walk-forward tuning: each year in CV_YEARS is validated on a model trained on all earlier years (purged by H)
CV_YEARS = (2023, 2024)
GRID = dict(max_depth=(3, 4, 6), min_child_weight=(1, 10), lr_n=((0.05, 400), (0.03, 600)))

EXTRA_DROP = []       # extra columns to exclude from X

DL_VERDICT = ("dl_validation gate: no target passed (rain_occ: DL 0.208 vs XGBoost 0.236; "
              "t2m_min: DL 0.165 vs XGBoost 0.190) -> MACHINE LEARNING SUFFICIENT, "
              "so the XGBoost models are the ones explained. Note: features (domain-wide precipitation, "
              "trends, dewpoint) and hyper-parameters (walk-forward tuning) were updated after this gate; "
              "the gate numbers refer to the earlier pilot and were not re-run.")