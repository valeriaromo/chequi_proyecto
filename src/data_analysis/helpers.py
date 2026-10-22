from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import calendar

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parents[1]  # chequi_proyecto

DATASET_VERSION = 1.0
LAB_OBJECTIVE = "Exploratory data analysis of the master table, with focus on the pilot target T2M_MIN and its relationship with precipitation and AOD."
TEAM_MEMBERS = ["Valeria", "Fany", "Lepipeline"]
DATASET_SOURCE = ("master_table.parquet built by src/pipeline/build_master_table.py""(NASA POWER, CHIRPS, MAIAC AOD, INEGI physical attributes)")

# variables
FLAGS = ["qc_aod_imputed", "qc_aod_pocos_pixeles"]
TEMPORAL = ["date", "month", "doy_sin", "doy_cos"]
STATIC_GEO = ["elevacion_msnm", "distancia_rio_m", "geom_centroid_lat", "geom_centroid_lon"]
BASE_WEATHER = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PS","PRECTOTCORR", "precipitation_mm"]
CORE = ["T2M_MAX", "T2M_MIN", "T2M", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PS", "PRECTOTCORR","precipitation_mm", "aod_055_mean", "aod_qa_best_pct", "aod_055_count", "t2m_rango","vpd_kpa", "dias_sin_lluvia", "rain_30d"]
UNITS = {"T2M": "°C", "T2M_MAX": "°C", "T2M_MIN": "°C", "t2m_rango": "°C", "RH2M": "%",
         "WS2M": "m/s", "PS": "kPa", "vpd_kpa": "kPa", "precipitation_mm": "mm",
         "PRECTOTCORR": "mm", "rain_3d": "mm", "rain_7d": "mm", "rain_30d": "mm",
         "rain_90d": "mm", "dias_sin_lluvia": "days", "aod_qa_best_pct": "%",
         "elevacion_msnm": "m", "distancia_rio_m": "m"}
RELEVANCE = {
    "T2M_MIN": "Pilot target (1-day horizon); its shape sets the error scale for the baselines.",
    "T2M_MAX": "Candidate target excluded from the pilot; its distribution is the reference for that decision.",
    "precipitation_mm": "Basis of the rain target; zero-inflation drives the choice of occurrence vs amount modelling.",
    "aod_055_mean": "Satellite aerosol signal; its distribution and gaps determine whether it can be used as a feature.",
    "aod_qa_best_pct": "Quality indicator for AOD retrievals.",
}
WET_MM = 1.0
FIG_DIR: Path = HERE / "output" / "figures"
TAB_DIR: Path = HERE / "output" / "tables"
FACTS: dict = {}
REG: list = []

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


# ===========================================================================
# Small helpers
# ===========================================================================
def fmt(x, nd=2):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return "n/a"
    ax = abs(float(x))
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax == 0:
        return "0"
    if ax < 0.01:
        return f"{x:.4f}"
    return f"{x:.{nd}f}"


def unit(c):
    return f" {UNITS[c]}" if c in UNITS else ""


def skew_label(g):
    a = abs(g)
    if a < 0.5:
        return "approximately symmetric"
    return f"{'moderately' if a < 1 else 'strongly'} {'right' if g > 0 else 'left'}-skewed"


def is_discrete(s):
    return pd.api.types.is_integer_dtype(s) or s.nunique() <= 40


def is_static(df, c):
    return df.groupby("municipio")[c].nunique(dropna=True).max() == 1


def group_of(c):
    if c in STATIC_GEO:
        return "Static geographic"
    if c.startswith("aod"):
        return "AOD"
    if c in BASE_WEATHER:
        return "Base weather"
    return "Derived"


def quant_vars(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in FLAGS + TEMPORAL]


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    return name


def reg(step, file, title, compared, obs, relevance):
    REG.append(dict(step=step, file=file, title=title, compared=compared, obs=obs,
                    relevance=relevance))


def hbox(ax, x):
    try:
        return ax.boxplot(x, orientation="horizontal", flierprops=dict(markersize=2, alpha=.4))
    except TypeError:
        return ax.boxplot(x, vert=False, flierprops=dict(markersize=2, alpha=.4))


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append(fmt(v))
            else:
                cells.append(str(v).replace("|", "/").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def full_years(d):
    yrs = []
    for y, g in d.groupby("yr"):
        need = 366 if calendar.isleap(int(y)) else 365
        if (g.groupby("municipio").size() == need).all():
            yrs.append(int(y))
    return yrs


def distinct_series(d, col):
    wide = d.pivot(index="date", columns="municipio", values=col)
    return int(wide.round(3).T.drop_duplicates().shape[0]), int(wide.shape[1])


def annual_rain_by_muni(d):
    full = full_years(d)
    if full:
        s = (d[d.yr.isin(full)].groupby(["municipio", "yr"])["precipitation_mm"].sum()
             .groupby("municipio").mean())
        return s, f"mean annual total over full years {full[0]}-{full[-1]}"
    return d.groupby("municipio")["precipitation_mm"].mean() * 365.25, "mean daily x 365.25"

