import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "pipeline" / "deep_learning"))   # reuse params, data, model, train, evaluate

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from deep_learning import params as P

OUT_DIR = HERE / "outputs"
FIG_DIR = OUT_DIR / "figures"
FASE2 = P.FASE2_DIR
SEEDS = [42, 43, 44, 45, 46]

# one single target config per target and one multitask over all targets (scales when TARGETS grows)
CONFIGS = {f"{t}_only": [t] for t in P.TARGETS} | {"multitask": list(P.TARGETS)}
CLIM_WINDOW = 15            # half width (days) of the circular smoothing of the doy climatology
SKILL_METRIC = "mae"        # regression skill = 1 - err(DL)/err(best baseline) 

# XGBoost frozen results. filters select h=1 / global rows; the loader fails loudly if a column is missing.
BASELINE = {
    "rain_occ": dict(csv=FASE2 / "rain_eval.csv", filters={"h": 1}, metric="bss"),               
    "t2m_min": dict(csv=FASE2 / "scope_T2M_MIN.csv", filters={"h": 1}, metric="skill_global"),  
}

NOISE_K = 2.0               # DL must beat XGBoost by > NOISE_K * seed std 
MIN_GAIN = 0.02             # and by at least this much, in every fold

# placebo test: shuffle the target and check that the model cannot learn it (BSS < PLACEBO_MAX, R2 < PLACEBO_MAX_R2)
PLACEBO_YEAR = 2025
PLACEBO_MAX = 0.02          # shuffled target BSS (clf) must stay below this
PLACEBO_MAX_R2 = 0.05       # shuffled target R2 (reg) must stay below this

XAI_YEAR = 2025
XAI_SEED = SEEDS[0]
PERM_REPEATS = 5
TOP_K = 10
LAG_BLOCKS = [(1, 3), (4, 7), (8, 14), (15, 30)]            # lags in days before the forecast origin

# Group permutation handles collinear lags/rolling features. Substring match, case-insensitive; empty matches skipped.
FEATURE_GROUPS = {
    "temperatura": ["t2m"], "lluvia": ["precip", "rain", "lluvia"], "humedad": ["rh2m", "vpd", "vapor"],
    "viento": ["ws2m"], "presion": ["ps_", "pressure", "presion"], "radiacion": ["rad", "allsky", "sw_"],
    "estacionalidad": ["sin", "cos", "doy", "season"],
}
FORCE_XAI = False           # True: explain the pilot targets even if none passes the gate (diagnostic only)