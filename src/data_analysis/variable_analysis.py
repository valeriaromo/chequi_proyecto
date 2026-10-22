from __future__ import annotations
import argparse
import inspect
import sys
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parents[1]
from helpers import fmt, md_table, DATASET_VERSION, TEAM_MEMBERS, DATASET_SOURCE

LAB_OBJECTIVE = ("Exploratory data analysis of the master table, with focus on the pilot target T2M_MIN and its relationship with precipitation and AOD.")
TARGET, NEXT = "T2M_MIN", "T2M_MIN_next"
SEL = ["T2M_MIN", "T2M_MAX", "T2M", "t2m_rango", "RH2M", "vpd_kpa", "WS2M", "ALLSKY_SFC_SW_DWN","PS", "precipitation_mm", "PRECTOTCORR", "rain_30d", "dias_sin_lluvia", "aod_055_mean"]
NASA_COLS = ["T2M", "T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN", "PS", "PRECTOTCORR"]
STATIC_VARS = ["elevacion_msnm", "distancia_rio_m"]
UNITS = {"T2M": "°C", "T2M_MAX": "°C", "T2M_MIN": "°C", "T2M_MIN_next": "°C", "t2m_rango": "°C","RH2M": "%", "WS2M": "m/s", "PS": "kPa", "vpd_kpa": "kPa", "precipitation_mm": "mm","PRECTOTCORR": "mm", "rain_30d": "mm", "dias_sin_lluvia": "days","elevacion_msnm": "m", "distancia_rio_m": "m"}
PROJECT_RELEVANCE = {
    "T2M_MIN": (1.0, "pilot target; its own memory is the strongest practical signal"),
    "T2M_MAX": (0.7, "co-moving temperature signal (T2M_MAX itself failed the pilot)"),
    "T2M": (0.6, "mean temperature, near-redundant with the extremes"),
    "t2m_rango": (0.6, "diurnal range, a cloud/dryness proxy derived from the target"),
    "RH2M": (0.8, "moisture controls nocturnal radiative cooling"),
    "vpd_kpa": (0.6, "derived from temperature and humidity"),
    "WS2M": (0.5, "mixing of the nocturnal boundary layer"),
    "ALLSKY_SFC_SW_DWN": (0.7, "radiation / cloudiness"),
    "PS": (0.5, "synoptic pressure signal"),
    "precipitation_mm": (0.9, "lab objective focus and basis of the rain target"),
    "PRECTOTCORR": (0.6, "second rainfall product"),
    "rain_30d": (0.7, "antecedent moisture"),
    "dias_sin_lluvia": (0.6, "dry-spell length"),
    "aod_055_mean": (0.7, "lab objective focus, but 20% missing and gaps are seasonal"),
    "elevacion_msnm": (0.5, "static; only separates municipalities"),
    "distancia_rio_m": (0.4, "static; only separates municipalities"),
}
STRENGTH_BINS = [(0.2, "Very Weak"), (0.4, "Weak"), (0.6, "Moderate"), (0.8, "Strong")]
PRACTICAL_HIGH, PRACTICAL_MED = 0.40, 0.20   # diferentes correlaciones speerman para considerar que una relación es de alta o mediana relevancia práctica
NO_REL = 0.10    # si rho < 0.10 no hay relación aparente
YEAR_CONSISTENCY = 0.80
BOOT_B, BOOT_L = 500, 30
FIG_DIR: Path = HERE / "output" / "variable_analysis" / "figures"
TAB_DIR: Path = HERE / "output" / "variable_analysis" / "tables"
FACTS: dict = {}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25,"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
HYPOTHESES = [
    ("H1", "T2M_MAX", TARGET, "+", "Warm days and warm nights share seasonal and synoptic drivers."),
    ("H2", TARGET, NEXT, "+", "Night-time temperature has strong day-to-day persistence."),
    ("H3", "RH2M", TARGET, "+", "Humid air limits radiative cooling at night."),
    ("H4", "RH2M", "t2m_rango", "-", "Humid, cloudier air damps the diurnal temperature range."),
    ("H5", "ALLSKY_SFC_SW_DWN", "T2M_MAX", "+", "More incoming radiation raises the daytime maximum."),
    ("H6", "precipitation_mm", "T2M_MAX", "-", "Rainy days are cloudier and cooler during the day."),
    ("H7", "precipitation_mm", TARGET, "0", "Clouds warm nights while evaporation cools them: net effect may be small."),
    ("H8", "PRECTOTCORR", "precipitation_mm", "+", "Two products measuring the same rainfall."),
    ("H9", "rain_30d", "RH2M", "+", "Recent rain moistens the air."),
    ("H10", "rain_30d", "aod_055_mean", "-", "Recent rain washes out aerosols and suppresses dust."),
    ("H11", "aod_055_mean", TARGET, "0", "Column aerosol is weakly tied to surface night temperature."),
    ("H12", "dias_sin_lluvia", "T2M_MAX", "+", "Long dry spells coincide with hotter afternoons."),
    ("H13", "WS2M", TARGET, "0", "Wind may matter little for night minima at daily scale."),
    ("H14", "PS", TARGET, "0", "Pressure level alone should say little about the minimum temperature."),
]
STATIC_HYPOTHESES = [
    ("H15", "elevacion_msnm", "T2M", "-", "Higher municipalities should be cooler (lapse rate)."),
    ("H16", "distancia_rio_m", "RH2M", "-", "Municipalities closer to rivers may be more humid."),
]
DERIVED_PAIRS = [{"T2M", "T2M_MAX"}, {"T2M", TARGET}, {"T2M_MAX", "t2m_rango"}, {TARGET, "t2m_rango"},
                 {"T2M", "t2m_rango"}, {"precipitation_mm", "rain_30d"}]


def unit(c):
    return f" {UNITS[c]}" if c in UNITS else ""

def lab(c):
    return "T2M_MIN (t+1)" if c == NEXT else c

def cf(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"

def by_construction(x, y):
    if {x, y} in DERIVED_PAIRS:
        return True
    return "vpd_kpa" in (x, y) and bool({x, y} & {"RH2M", "T2M", "T2M_MAX", TARGET})

def strength(v):
    a = abs(v)
    for thr, name in STRENGTH_BINS:
        if a < thr:
            return name
    return "Very Strong"

def direction(rho, lo, hi):
    if abs(rho) < NO_REL or lo <= 0 <= hi:
        return "No Apparent Relationship"
    return "Positive" if rho > 0 else "Negative"

def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    return name

def distinct_series(d, col):
    wide = d.pivot(index="date", columns="municipio", values=col)
    return int(wide.round(3).T.drop_duplicates().shape[0]), int(wide.shape[1])

def load_data(path):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["municipio"] = df["municipio"].astype(str)
    df = df.sort_values(["municipio", "date"]).reset_index(drop=True)
    df["yr"] = df["date"].dt.year
    dates = np.sort(df["date"].unique())
    df["dcode"] = np.searchsorted(dates, df["date"].values)
    df[NEXT] = df.groupby("municipio")[TARGET].shift(-1)
    return df, len(dates)

def build_anomalies(df, cols, win=15):
    doy = df["date"].dt.dayofyear.values
    munis = pd.Index(sorted(df["municipio"].unique()))
    mrow = munis.get_indexer(df["municipio"])
    A = df[["municipio", "date", "yr", "dcode"]].copy()
    for c in cols:
        prof = (df.groupby(["municipio", df["date"].dt.dayofyear])[c].mean().unstack()
                .reindex(index=munis, columns=range(1, 367)))
        ext = pd.concat([prof.iloc[:, -win:], prof, prof.iloc[:, :win]], axis=1)
        sm = ext.T.rolling(win, center=True, min_periods=1).mean().T.iloc[:, win:-win]
        A[c] = df[c].values - sm.values[mrow, doy - 1]
    A[NEXT] = A.groupby("municipio")[TARGET].shift(-1)
    return A

class BlockBoot:
    def __init__(self, T, B=BOOT_B, L=BOOT_L, seed=42): # se bloquea el bootstrap para preservar la estructura temporal de los datos
        rng = np.random.default_rng(seed)
        self.T, self.L = T, L
        self.starts = rng.integers(0, T - L + 1, size=(B, int(np.ceil(T / L))))
        self.n_blocks = int(np.ceil(T / L))

    def suff(self, dc, x, y):
        T = self.T
        return np.column_stack([np.bincount(dc, minlength=T).astype(float),np.bincount(dc, x, T), np.bincount(dc, y, T),np.bincount(dc, x * x, T), np.bincount(dc, y * y, T),np.bincount(dc, x * y, T)])
    
    @staticmethod
    def _rs(S):
        n, sx, sy, sxx, syy, sxy = [S[..., i] for i in range(6)]
        cxy, vx, vy = sxy - sx * sy / n, sxx - sx * sx / n, syy - sy * sy / n
        with np.errstate(all="ignore"):
            return cxy / np.sqrt(vx * vy), cxy / vx

    def estimate(self, S):
        C = np.vstack([np.zeros((1, 6)), np.cumsum(S, axis=0)])
        Sb = (C[self.starts + self.L] - C[self.starts]).sum(axis=1)
        rb, bb = self._rs(Sb)
        r, b = self._rs(S.sum(axis=0))
        return float(r), float(b), np.nanpercentile(rb, [2.5, 97.5]), np.nanpercentile(bb, [2.5, 97.5])


def pair_stats(F, x, y, bb):
    sub = F[[x, y, "dcode"]].dropna()
    if len(sub) < 100:
        return None
    xv, yv, dc = sub[x].to_numpy(float), sub[y].to_numpy(float), sub["dcode"].to_numpy()
    if xv.std() == 0 or yv.std() == 0:
        return None
    r, b, rci, bci = bb.estimate(bb.suff(dc, xv, yv))
    rho, _, rhoci, _ = bb.estimate(bb.suff(dc, stats.rankdata(xv), stats.rankdata(yv)))
    n = len(sub)
    p = 2 * stats.t.sf(abs(r * np.sqrt((n - 2) / max(1 - r * r, 1e-12))), n - 2)
    return dict(n=n, r=r, r_lo=rci[0], r_hi=rci[1], rho=rho, rho_lo=rhoci[0], rho_hi=rhoci[1],slope=b, slope_lo=bci[0], slope_hi=bci[1], intercept=yv.mean() - b * xv.mean(),r2=r * r, p_naive=p, sd_x=xv.std(), sd_y=yv.std())

def year_consistency(F, x, y, sign):
    sub = F[[x, y, "yr"]].dropna()
    vals = [g[x].corr(g[y], method="spearman") for _, g in sub.groupby("yr")
            if len(g) >= 200 and g[x].std() > 0 and g[y].std() > 0]
    if not vals:
        return np.nan, 0
    return float((np.sign(np.array(vals)) == sign).mean()), len(vals)

def interpret_pair(rec):
    rho, rhoa = rec["rho"], rec["rho_a"]
    dr = direction(rho, rec["rho_lo"], rec["rho_hi"])
    da = direction(rhoa, rec["rho_a_lo"], rec["rho_a_hi"])
    sr, sa = strength(rho), strength(rhoa)
    if dr == da == "No Apparent Relationship":
        t = f"No apparent monotone relationship, raw (rho {cf(rho)}) or deseasonalised (rho {cf(rhoa)})."
    elif da == "No Apparent Relationship":
        t = (f"{sr} {dr.lower()} raw association (rho {cf(rho)}) that vanishes after removing the annual "
             f"cycle (rho {cf(rhoa)}): the co-movement is seasonal.")
    elif dr == "No Apparent Relationship":
        t = (f"Masked by seasonality: no raw association (rho {cf(rho)}) but a {sa.lower()} {da.lower()} "
             f"one once deseasonalised (rho {cf(rhoa)}).")
    elif np.sign(rho) != np.sign(rhoa):
        t = f"Sign reverses after removing seasonality (raw {cf(rho)}, deseasonalised {cf(rhoa)})."
    else:
        keep = "persists beyond the annual cycle" if abs(rhoa) >= .5 * abs(rho) else "is largely seasonal"
        t = (f"{sr} {dr.lower()} raw (rho {cf(rho)}); {sa.lower()} deseasonalised (rho {cf(rhoa)}): "
             f"the relationship {keep}.")
    if abs(rho - rec["r"]) > .1:
        t += f" Pearson ({cf(rec['r'])}) and Spearman differ: non-linear or driven by zeros/extremes."
    if rec["derived"]:
        t += " Partly by construction (derived variables)."
    return t

def evidence_level(rec):
    c1 = rec["rho_a_lo"] > 0 or rec["rho_a_hi"] < 0  # test de hipótesis para ver si la correlación deseasonalizada es significativamente diferente de cero
    c2 = rec["r_a_lo"] > 0 or rec["r_a_hi"] < 0 # igual para Pearson
    c3 = (rec["year_consist"] >= YEAR_CONSISTENCY) if not np.isnan(rec["year_consist"]) else False
    c4 = abs(rec["rho_a"]) >= NO_REL    # descarta relaciones muy débiles
    if c1 and c2 and c3 and c4:
        return "High"
    if c1 and (c3 or c4):
        return "Medium"
    return "Low"

def practical_level(rec):
    a = abs(rec["rho_a"])
    return "High" if a >= PRACTICAL_HIGH else "Medium" if a >= PRACTICAL_MED else "Low"

def nature(rec):
    ar, aa = abs(rec["rho"]), abs(rec["rho_a"])
    if aa >= PRACTICAL_MED:
        return "Beyond seasonal"
    if ar >= PRACTICAL_MED:
        return "Seasonal co-movement"
    return "Weak / none"

def verdict(exp, rec):
    ex_a = rec["rho_a_lo"] > 0 or rec["rho_a_hi"] < 0
    ex_r = rec["rho_lo"] > 0 or rec["rho_hi"] < 0
    if exp == "0":
        if abs(rec["rho_a"]) < PRACTICAL_MED:
            return "Supported" if abs(rec["rho"]) < PRACTICAL_MED else "Supported beyond the seasonal cycle"
        return "Not supported"
    s = 1 if exp == "+" else -1
    if ex_a and np.sign(rec["rho_a"]) != s:
        return "Contradicted"
    if ex_a and abs(rec["rho_a"]) >= NO_REL:
        return "Supported"
    if ex_r and np.sign(rec["rho"]) == s:
        return "Supported only through seasonality"
    return "Inconclusive"

def analyse_registry(df, A, bb):
    pairs, seen = [], {}
    for hid, x, y, exp, why in HYPOTHESES:
        pairs.append(dict(x=x, y=y, kind="hypothesis", tag=hid, expected=exp, why=why))
        seen[(x, y)] = True
    for v in SEL:
        for tgt, kind in ((TARGET, "same_day"), (NEXT, "next_day")):
            if v != tgt and (v, tgt) not in seen:
                pairs.append(dict(x=v, y=tgt, kind=kind, tag="", expected="", why=""))
                seen[(v, tgt)] = True
    out = []
    for p in pairs:
        raw = pair_stats(df, p["x"], p["y"], bb)
        an = pair_stats(A, p["x"], p["y"], bb)
        if raw is None or an is None:
            continue
        rec = dict(p)
        rec.update(raw)
        rec.update({k + "_a" if k in ("rho", "slope", "intercept", "r2", "sd_x", "sd_y", "n") else
                    k.replace("r_lo", "r_a_lo").replace("r_hi", "r_a_hi").replace("rho_lo", "rho_a_lo").replace("rho_hi", "rho_a_hi").replace("slope_lo", "slope_a_lo").replace("slope_hi", "slope_a_hi").replace("p_naive", "p_naive_a")
                    if k not in ("r",) else "r_a": v for k, v in an.items()})
        rec["derived"] = by_construction(p["x"], p["y"])
        rec["year_consist"], rec["n_years"] = year_consistency(A, p["x"], p["y"], np.sign(rec["rho_a"]))
        rec["strength_raw"], rec["strength_a"] = strength(rec["rho"]), strength(rec["rho_a"])
        rec["dir_raw"] = direction(rec["rho"], rec["rho_lo"], rec["rho_hi"])
        rec["dir_a"] = direction(rec["rho_a"], rec["rho_a_lo"], rec["rho_a_hi"])
        rec["interpretation"] = interpret_pair(rec)
        rec["evidence"], rec["practical"], rec["nature"] = evidence_level(rec), practical_level(rec), nature(rec)
        if p["expected"]:
            rec["verdict"] = verdict(p["expected"], rec)
        out.append(rec)
    return pd.DataFrame(out)

def static_hypotheses(df):
    rows = []
    for hid, x, y, exp, why in STATIC_HYPOTHESES:
        m = df.groupby("municipio").agg(xv=(x, "first"), yv=(y, "mean"))
        n_dist, _ = distinct_series(df, y)
        r, p = stats.pearsonr(m.xv, m.yv)
        rho = stats.spearmanr(m.xv, m.yv)[0]
        z, se = np.arctanh(r), 1 / np.sqrt(len(m) - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
        s = 1 if exp == "+" else -1
        verdict_ = ("Inconclusive (effective n too small)" if n_dist < 10 else
                    "Supported" if lo > 0 and s > 0 or hi < 0 and s < 0 else "Inconclusive")
        rows.append(dict(tag=hid, x=x, y=f"{y} (municipality mean)", expected=exp, n=len(m),
                         distinct_series=n_dist, r=r, rho=rho, ci_lo=lo, ci_hi=hi, p=p,
                         verdict=verdict_))
    return pd.DataFrame(rows)

# FIGURAS
def heatmap(M, title, name):
    cols = list(M.columns)
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(M.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels([lab(c) for c in cols], rotation=70, ha="right", fontsize=7)
    ax.set_yticklabels([lab(c) for c in cols], fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, cf(M.values[i, j]), ha="center", va="center", fontsize=6)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("Variable")
    ax.set_ylabel("Variable")
    fig.colorbar(im, ax=ax, label="Correlation")
    return save_fig(fig, name)

def dashboard(df, reg):
    hyp = reg[reg.kind == "hypothesis"].reset_index(drop=True)
    rows, cols = int(np.ceil(len(hyp) / 4)), 4
    fig, axes = plt.subplots(rows, cols, figsize=(17, 3.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    obs = []
    for ax, (_, r) in zip(axes, hyp.iterrows()):
        sub = df[[r.x, r.y]].dropna()
        xv, yv = sub[r.x].to_numpy(float), sub[r.y].to_numpy(float)
        xl, xh = np.quantile(xv, [.001, .999])
        yl, yh = np.quantile(yv, [.001, .999])
        ax.hexbin(np.clip(xv, xl, xh), np.clip(yv, yl, yh), gridsize=32, bins="log", mincnt=1, cmap="viridis")
        xs = np.linspace(xl, xh, 20)
        ax.plot(xs, r.intercept + r.slope * xs, color="#E45756", lw=1.3,
                label=f"slope {r.slope:+.3g}; rho {cf(r.rho)} (deseas. {cf(r.rho_a)})")
        ax.set_ylim(yl, yh)
        ax.set_title(f"{r.tag}: {r.x} vs {lab(r.y)}", fontsize=8)
        ax.set_xlabel(f"{r.x}{unit(r.x)}", fontsize=7)
        ax.set_ylabel(f"{lab(r.y)}{unit(r.y)}", fontsize=7)
        ax.legend(fontsize=6, loc="best")
        obs.append(dict(Chart=r.tag, Variables=f"{r.x} vs {lab(r.y)}", Main_observations=(f"{r.strength_raw} raw (rho {cf(r.rho)}), {r.strength_a.lower()} " f"deseasonalised (rho {cf(r.rho_a)}); {r.nature.lower()}; expected "f"'{r.expected}' -> {r.verdict.lower()}."), Analytical_relevance=r.why))
    for ax in axes[len(hyp):]:
        ax.axis("off")
    fig.suptitle("Relationship dashboard: one panel per hypothesis (hexbin density, log colour)",
                 fontweight="bold")
    return save_fig(fig, "fig_step4_relationship_dashboard.png"), pd.DataFrame(obs)

def driver_chart(reg):
    d = reg[(reg.y == NEXT)].copy()
    d = d.reindex(d.rho_a.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    pos = np.arange(len(d))
    ax.barh(pos + .2, d.rho, height=.38, color="#9ecae1", label="Raw Spearman")
    ax.barh(pos - .2, d.rho_a, height=.38, color="#3182bd", label="Deseasonalised Spearman",
            xerr=[d.rho_a - d.rho_a_lo, d.rho_a_hi - d.rho_a], error_kw=dict(lw=.8, capsize=2))
    ax.set_yticks(pos)
    ax.set_yticklabels(d.x)
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("Spearman correlation with T2M_MIN at t+1 (bars: 95% block-bootstrap CI)")
    ax.set_ylabel("Variable measured at day t")
    ax.set_title("Which variables relate to tomorrow's minimum temperature?")
    ax.legend(loc="lower right")
    return save_fig(fig, "fig_step4_driver_correlations.png")

# REGRESIÓN
def select_models(reg, k=8, forced=(("precipitation_mm", NEXT), ("aod_055_mean", NEXT))):
    c = reg[~reg.derived].copy()
    c["score"] = np.maximum(c.r2, c.r2_a)
    top = c.sort_values("score", ascending=False).head(k)
    for key in forced:
        row = c[(c.x == key[0]) & (c.y == key[1])]
        if len(row) and not ((top.x == key[0]) & (top.y == key[1])).any():
            top = pd.concat([top, row])
    return top.reset_index(drop=True)

def regression_matrix(models):
    rows = []
    for i, m in models.iterrows():
        eq = f"{lab(m.y)} = {m.intercept:.2f} {'+' if m.slope >= 0 else '-'} {abs(m.slope):.3g}*{m.x}"
        eff = m.slope_a * m.sd_x_a
        use = "High" if m.r2_a >= .25 else "Medium" if m.r2_a >= .05 else "Low"
        interp = (f"Each +1{unit(m.x) or ' unit'} of {m.x} goes with {m.slope:+.3g}{unit(m.y)} in {lab(m.y)}. "
                  f"Explains {m.r2 * 100:.1f}% of variance raw but {m.r2_a * 100:.1f}% once seasonality is removed; "
                  f"+1 SD of the anomaly of {m.x} ≈ {eff:+.2f}{unit(m.y)}. Analytical usefulness: {use}."
                  + (" OLS on a zero-inflated variable: descriptive only." if "precip" in m.x or "PRECTOT" in m.x
                     or "rain" in m.x else ""))
        rows.append(dict(Model=f"M{i + 1}", Independent_Variable=m.x, Dependent_Variable=lab(m.y),
                         Equation=eq, R2_raw=m.r2, R2_deseasonalised=m.r2_a,
                         Slope_95CI=f"{m.slope:+.3g} [{m.slope_lo:.3g}, {m.slope_hi:.3g}]",
                         Usefulness=use, Interpretation=interp))
    return pd.DataFrame(rows)

def graph_regression(df, m):
    sub = df[[m.x, m.y]].dropna()
    xv, yv = sub[m.x].to_numpy(float), sub[m.y].to_numpy(float)
    xl, xh = np.quantile(xv, [.001, .999])
    yl, yh = np.quantile(yv, [.001, .999])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.hexbin(np.clip(xv, xl, xh), np.clip(yv, yl, yh), gridsize=32, bins="log", mincnt=1, cmap="viridis")
    xs = np.linspace(xl, xh, 20)
    ax.plot(xs, m.intercept + m.slope * xs, color="#E45756", lw=1.3,
            label=f"slope {m.slope:+.3g}; rho {cf(m.rho)} (deseas. {cf(m.rho_a)})")
    ax.set_ylim(yl, yh)
    ax.set_title(f"{m.x} vs {lab(m.y)}", fontsize=8)
    ax.set_xlabel(f"{m.x}{unit(m.x)}", fontsize=7)
    ax.set_ylabel(f"{lab(m.y)}{unit(m.y)}", fontsize=7)
    ax.legend(fontsize=6, loc="best")
    return save_fig(fig, f"fig_step4_regression_{m.x}_vs_{m.y}.png")

# MATRIZ Y RANKING
def evidence_matrix(reg, models):
    keys = set(zip(reg[reg.kind == "hypothesis"].x, reg[reg.kind == "hypothesis"].y)) | \
        set(zip(models.x, models.y))
    d = reg[[(x, y) in keys for x, y in zip(reg.x, reg.y)]]
    rows = []
    for _, r in d.iterrows():
        rows.append(dict(Relationship=f"{r.x} -> {lab(r.y)}", Evidence_Level=r.evidence,Practical_Relevance=r.practical, Nature=r.nature,Rho_deseas_95CI=f"{cf(r.rho_a)} [{cf(r.rho_a_lo)}, {cf(r.rho_a_hi)}]",Pearson_deseas_95CI=f"{cf(r.r_a)} [{cf(r.r_a_lo)}, {cf(r.r_a_hi)}]",p_naive=r.p_naive_a, Years_same_sign=f"{r.year_consist * 100:.0f}% of {r.n_years}"
                         if not np.isnan(r.year_consist) else "n/a"))
    return pd.DataFrame(rows)
LEVEL = {"High": 1.0, "Medium": .5, "Low": 0.0}

def impact_ranking(df, reg, static_df):
    rows = []
    nxt = reg[reg.y == NEXT].set_index("x")
    for v in SEL:
        if v not in nxt.index:
            continue
        r = nxt.loc[v]
        c1, c2 = min(abs(r.rho_a) / .5, 1), min(r.r2_a / .25, 1)
        c3, c4, (c5, why) = LEVEL[r.evidence], LEVEL[r.practical], PROJECT_RELEVANCE[v]
        score = .30 * c1 + .20 * c2 + .20 * c3 + .10 * c4 + .20 * c5
        rows.append(dict(Variable=v, Score=score, Statistical_Evidence=(f"rho deseas. {cf(r.rho_a)} [{cf(r.rho_a_lo)}, {cf(r.rho_a_hi)}], "f"raw {cf(r.rho)}; R2 deseas. {r.r2_a * 100:.1f}%; evidence {r.evidence}"), Practical_Relevance=f"{r.practical} practical effect; {r.nature.lower()}; project: {why}",Autoregressive=(v == TARGET)))
    for _, s in static_df.iterrows():
        v = s.x
        rng_ = df.groupby("municipio")["T2M"].mean()
        spread = float(rng_.max() - rng_.min())
        c1, c2 = min(abs(s.rho) / .5, 1), min(s.r ** 2 / .25, 1)
        c3 = 0.0 if s.distinct_series < 10 else (0.5 if s.p < .05 else 0.0)
        c4 = 1.0 if spread >= 3 else .5 if spread >= 1 else 0.0
        c5, why = PROJECT_RELEVANCE[v]
        score = .30 * c1 + .20 * c2 + .20 * c3 + .10 * c4 + .20 * c5
        rows.append(dict(Variable=v, Score=score,Statistical_Evidence=(f"municipality level: rho {cf(s.rho)}, r {cf(s.r)} "f"[{cf(s.ci_lo)}, {cf(s.ci_hi)}], n={s.n} but only {s.distinct_series} "f"distinct {s.y.split(' ')[0]} series"),Practical_Relevance=(f"municipal mean T2M spread {spread:.1f} °C; project: {why}"),Autoregressive=False))
    out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    out["Impact_Level"] = np.where(out.Score >= .60, "High Impact",
                                   np.where(out.Score >= .35, "Medium Impact", "Low Impact"))
    out.insert(0, "Rank", range(1, len(out) + 1))
    return out

# REPORTE
def md_src(*fns):
    return "```python\n" + "\n".join(inspect.getsource(f) for f in fns) + "```"

def classify(rank, reg):
    non_auto = rank[~rank.Autoregressive & ~rank.Variable.isin(STATIC_VARS)]
    n_high = int((non_auto.Impact_Level == "High Impact").sum())
    nxt = reg[(reg.y == NEXT) & (reg.x != TARGET)]
    best = float(nxt.rho_a.abs().max())
    n_beyond = int((nxt.rho_a.abs() >= PRACTICAL_MED).sum())
    if n_high >= 3:
        label = "READY FOR PATTERN DISCOVERY"
    elif n_beyond == 0:
        label = "ADDITIONAL DATA REQUIRED"
    else:
        label = "CONDITIONALLY READY"
    why = (f"{n_high} non-autoregressive variable(s) reach High Impact (needed: 3 for READY); "
           f"{n_beyond} of {len(nxt)} predictors have |deseasonalised rho| >= {PRACTICAL_MED:.2f} with T2M_MIN(t+1) "
           f"(strongest {best:.2f}); a label of ADDITIONAL DATA REQUIRED would apply only if that count were 0.")
    return label, why, n_high, n_beyond, best

def write_variable_analysis_report(out_dir, df, reg, hyp_tbl, dash_tbl, models, regm, evid, rank, static_df, label, why,
                 n_dist, files):
    nasa = FACTS["distinct_nasa"]
    aod_miss = float(df["aod_055_mean"].isna().mean() * 100)
    aod_const = "aod_qa_best_pct" in df and df["aod_qa_best_pct"].nunique() <= 1
    outc = hyp_tbl.copy()
    top = rank[~rank.Autoregressive].head(5)
    strongest = reg[~reg.derived].sort_values("rho_a", key=np.abs, ascending=False).head(5)
    nonlin = reg[(reg.rho - reg.r).abs() > .1]
    verdicts = reg[reg.kind == "hypothesis"].verdict.value_counts().to_dict()
    L = ["# Data Intelligence Lab - Relationship Analysis (Steps 2 to 10)", "",
         "## Dataset documentation", "",
         f"- **Dataset source:** {DATASET_SOURCE}", f"- **Dataset version:** {DATASET_VERSION}", f"- **Date of analysis:** {date.today().isoformat()}", f"- **Objective:** {LAB_OBJECTIVE}",f"- **Team members:** {TEAM_MEMBERS}",f"- **Rows / municipalities / days:** {len(df):,} / {df.municipio.nunique()} / {n_dist:,}", "","Method note: every relationship is measured raw and deseasonalised (anomaly from the smoothed "
         "day-of-year mean of each municipality); confidence intervals come from a 30-day moving-block ""bootstrap over dates, because rows are neither independent across days nor across municipalities.", "",
         "## Step 2 - Relationship hypotheses", "", md_table(hyp_tbl[["Hypothesis", "Variables", "Expected_Relationship", "Rationale"]]), "",
         "## Step 3 - Correlation analysis", "","**Code used:**", "", md_src(build_anomalies, BlockBoot, pair_stats, strength, direction, interpret_pair), "","### Correlation matrices", "","Full matrices: `tables/step3_corr_*.csv`.", "",f"![Spearman raw](figures/{files['h_raw']})", "", f"![Spearman deseasonalised](figures/{files['h_an']})", "", "### Correlation analysis matrix", "",
         md_table(files["pair_tbl"]), ""]
    L += [f"Pairs where Pearson and Spearman differ by more than 0.10 (non-linear or zero/extreme driven): "f"{', '.join(f'{a}->{lab(b)}' for a, b in zip(nonlin.x, nonlin.y)) or 'none'}.", "",
          "## Step 4 - Relationship visualisation", "",f"![Relationship dashboard](figures/{files['dash']})", "",md_table(dash_tbl), "", f"![Driver correlations](figures/{files['drv']})", "","**Variables compared:** every predictor at day t against T2M_MIN at t+1. **Main observations:** "f"the strongest deseasonalised links are {', '.join(f'{a}->{lab(b)} ({cf(c)})' for a, b, c in zip(strongest.x, strongest.y, strongest.rho_a))}. ""**Analytical relevance:** the gap between light and dark bars is the share of each relationship that is only "
          "the annual cycle, which a climatology baseline already captures.", "",
          "## Step 5 - Regression analysis", "", "**Code used:**", "", md_src(select_models, regression_matrix), "","Models are the top relationships by variance explained (raw or deseasonalised), derived-by-construction pairs excluded, plus the two variables named in the lab objective (precipitation and AOD).", "",
          md_table(regm), "",
          "## Step 6 - Statistical evidence assessment", "", "**Code used:**", "", md_src(evidence_level, practical_level, nature, verdict), "",f"Criteria. Evidence = High if the block-bootstrap CIs of Spearman and Pearson (deseasonalised) both exclude 0, "
          f"the sign holds in at least {YEAR_CONSISTENCY * 100:.0f}% of years and |rho| >= {NO_REL}; Medium if the Spearman CI "f"excludes 0 and one of the other two holds; otherwise Low. Practical relevance is High/Medium/Low for |rho| "
          f">= {PRACTICAL_HIGH}/{PRACTICAL_MED}. Naive p-values (which treat all rows as independent) are shown only to "f"illustrate how overconfident they are.", "", md_table(evid), "","### Static variables (municipality level)", "", md_table(static_df), "",
          "## Step 7 - Variable impact ranking", "","Score = 0.30 correlation strength + 0.20 regression R2 + 0.20 evidence level + 0.10 practical relevance + ""0.20 project relevance (judgement weights in `PROJECT_RELEVANCE`). High >= 0.60, Medium >= 0.35.", "",
          md_table(rank[["Rank", "Variable", "Statistical_Evidence", "Practical_Relevance", "Impact_Level"]]), "",
          "## Step 8 - Variable impact assessment", "","### Relationship understanding", "",
          f"Of {len(reg[reg.kind == 'hypothesis'])} a-priori hypotheses, the outcomes are: "
          f"{', '.join(f'{v} x {k.lower()}' for k, v in verdicts.items())}. "
          f"Of {len(reg)} pairs examined, {int((reg.nature == 'Beyond seasonal').sum())} show a relationship beyond the "
          f"seasonal cycle, {int((reg.nature == 'Seasonal co-movement').sum())} are seasonal co-movement only and "
          f"{int((reg.nature == 'Weak / none').sum())} are weak or absent.", "",
          md_table(outc[["Hypothesis", "Variables", "Expected_Relationship"]].assign(
              Outcome=list(reg[reg.kind == "hypothesis"].verdict) + list(static_df.verdict))), "",
          "### Explanatory variables", "",
          "Most influential variables other than T2M_MIN's own memory: " +
          "; ".join(f"{r.Variable} ({r.Impact_Level.lower()}, score {r.Score:.2f})" for _, r in top.iterrows()) + ".", "",
          "### Analytical limitations", "",
          f"- NASA POWER series are not independent across municipalities: only {min(nasa.values())}-{max(nasa.values())} "f"distinct series among {df.municipio.nunique()} municipalities (previous analysis F2), so the effective sample "f"for NASA-based relationships is far below {len(df):,} rows; this is why bootstrap CIs by date block are used.",
          f"- AOD is missing in {aod_miss:.1f}% of rows with seasonal gaps (previous analysis F3), so AOD relationships "f"describe clear-sky days only." + (" `aod_qa_best_pct` is constant and carries no information." if aod_const else ""),"- Precipitation is zero-inflated (previous analysis F1); linear correlations and OLS slopes on rain are descriptive.","- Raw temperature relationships are dominated by the annual cycle (previous analysis F4); only the ""deseasonalised columns speak to non-seasonal signal, and the climatology used for that is estimated on the ""full sample (exploration, not forecasting).",
          "- Only simple (one-predictor) linear models were fitted; interactions, non-linearity and collinearity among ""predictors are not assessed. Correlation is not causation.","- Static variables explain differences between municipalities only and rest on 11 points with fewer distinct series.", "",
          "### Future analytical directions", "","Feature selection and machine learning on the variables ranked above; clustering of days into weather regimes (dry/wet, clear/cloudy) to separate regime-specific relationships; predictive analytics evaluated with the existing walk-forward folds and the smoothed-climatology baseline; a Pattern Discovery challenge focused on the non-linear pairs listed in Step 3.", "",f"### Project classification: **{label}**", "", why, "","## Step 9 - Variable impact assessment (summary)", "",
          md_table(pd.DataFrame([
              dict(Topic="Main explanatory variables", Summary="; ".join(f"{r.Variable} ({r.Impact_Level})" for _, r in rank.head(6).iterrows())),
              dict(Topic="Limitations", Summary=f"{min(nasa.values())}-{max(nasa.values())} distinct NASA series; AOD {aod_miss:.0f}% missing; "f"seasonality dominates raw correlations; simple linear models only"),
              dict(Topic="Opportunities", Summary=(f"{int((reg.nature == 'Beyond seasonal').sum())} relationships persist after "f"deseasonalising; strongest {strongest.iloc[0].x}->{lab(strongest.iloc[0].y)} "f"(rho {cf(strongest.iloc[0].rho_a)})")),
              dict(Topic="Readiness", Summary=label)])), "",
          "## Step 10 - Analytical roadmap", "","**Recommended next stage:** " + ("pattern discovery and feature selection." if label.startswith("READY") else "complete the open statistical work before pattern discovery: (1) fit a multivariable model on the deseasonalised ""anomalies to check which single-variable links survive jointly, (2) repeat the analysis by season, and ""(3) quantify the effect of the 3-series NASA grid on every pooled estimate.") + "", "","**Relationships and variables that deserve deeper investigation:**", ""]
    for _, r in rank[(rank.Impact_Level != "Low Impact") & ~rank.Autoregressive].head(6).iterrows():
        L.append(f"- {r.Variable}: {r.Statistical_Evidence}.")
    inc = reg[(reg.kind == "hypothesis") & reg.verdict.isin(["Inconclusive", "Not supported", "Contradicted"])]
    for _, r in inc.iterrows():
        L.append(f"- {r.tag} ({r.x} -> {lab(r.y)}): {r.verdict.lower()}; revisit with regime- or season-specific analysis.")
    L += ["", "## Executive Summary", "","**Strongest relationships (deseasonalised, excluding derived pairs):** " +"; ".join(f"{r.x} -> {lab(r.y)} (rho {cf(r.rho_a)}, 95% CI [{cf(r.rho_a_lo)}, {cf(r.rho_a_hi)}])"for _, r in strongest.iterrows()) + ".", "",
          "**Most influential variables:** " + "; ".join(f"{r.Variable} ({r.Impact_Level.lower()})" for _, r in rank.head(5).iterrows()) + ".", "",
          f"**Analytical limitations:** {min(nasa.values())}-{max(nasa.values())} distinct NASA series among "
          f"{df.municipio.nunique()} municipalities; AOD {aod_miss:.1f}% missing; zero-inflated rain; seasonality dominates raw "
          f"correlations; simple linear models only.", "",
          f"**Readiness for future activities: {label}.** {why}", ""]
    (out_dir / "variable_analysis_report.md").write_text("\n".join(L), encoding="utf-8")

def default_master_path():
    sys.path.insert(0, str(BASE_DIR))
    try:
        from config.settings import MASTER_TABLE_PATH
        return Path(MASTER_TABLE_PATH)
    except Exception:
        return BASE_DIR / "data" / "master" / "master_table.parquet"

def main():
    global FIG_DIR, TAB_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data_path = Path(a.data) if a.data else default_master_path()
    out_dir = Path(a.out) if a.out else HERE / "output" / "variable_analysis"
    FIG_DIR, TAB_DIR = out_dir / "figures", out_dir / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df, T = load_data(data_path)
    A = build_anomalies(df, SEL)
    bb = BlockBoot(T)
    FACTS["distinct_nasa"] = {c: distinct_series(df, c)[0] for c in NASA_COLS if c in df}
    print(f"Loaded {len(df):,} rows, {T} days; distinct NASA series: {FACTS['distinct_nasa']}")

    # Step 2
    hyp_tbl = pd.DataFrame([dict(Hypothesis=f"{h}: " + {"+": f"Increasing {x} may increase {lab(y)}","-": f"Increasing {x} may decrease {lab(y)}", "0": f"{x} may have little relationship with {lab(y)}"}[e],Variables=f"{x}, {lab(y)}", Expected_Relationship={"+": "Positive", "-": "Negative", "0": "Little / none"}[e], Rationale=w) for h, x, y, e, w in HYPOTHESES + STATIC_HYPOTHESES])
    hyp_tbl.to_csv(TAB_DIR / "step2_hypotheses.csv", index=False)

    # Step 3
    reg = analyse_registry(df, A, bb)
    static_df = static_hypotheses(df)
    VARS = SEL + [NEXT]
    M = {"pearson_raw": df[VARS].corr("pearson"), "spearman_raw": df[VARS].corr("spearman"),"spearman_anomaly": A[VARS].corr("spearman"), "pearson_anomaly": A[VARS].corr("pearson")}
    for k, v in M.items():
        v.to_csv(TAB_DIR / f"step3_corr_{k}.csv")
    files = dict(h_raw=heatmap(M["spearman_raw"], "Spearman correlation (raw)", "fig_step3_heatmap_spearman_raw.png"),h_an=heatmap(M["spearman_anomaly"], "Spearman correlation (deseasonalised)","fig_step3_heatmap_spearman_anomaly.png"))
    full = reg.assign(Variable_Pair=[f"{x} -> {lab(y)}" for x, y in zip(reg.x, reg.y)])
    full.drop(columns=["why"]).to_csv(TAB_DIR / "step3_correlation_pairs_full.csv", index=False)
    shown = pd.concat([reg[reg.kind == "hypothesis"],reg[(reg.kind == "next_day")].sort_values("rho_a", key=np.abs, ascending=False).head(10)])
    files["pair_tbl"] = pd.DataFrame(dict(
        Variable_Pair=[f"{x} -> {lab(y)}" for x, y in zip(shown.x, shown.y)], Pearson=shown.r.values,
        Spearman=shown.rho.values, Strength=shown.strength_raw.values, Direction=shown.dir_raw.values,
        Spearman_deseasonalised=shown.rho_a.values, Strength_deseas=shown.strength_a.values,
        Direction_deseas=shown.dir_a.values, Interpretation=shown.interpretation.values))

    # Step 4
    files["dash"], dash_tbl = dashboard(df, reg)
    files["drv"] = driver_chart(reg)
    dash_tbl.to_csv(TAB_DIR / "step4_relationship_dashboard.csv", index=False)

    # Step 5-7
    models = select_models(reg)
    regm = regression_matrix(models)
    regm.to_csv(TAB_DIR / "step5_regression_matrix.csv", index=False)
    for _, m in models.iterrows():
        graph_regression(df, m)
    evid = evidence_matrix(reg, models)
    evid.to_csv(TAB_DIR / "step6_evidence_matrix.csv", index=False)
    static_df.to_csv(TAB_DIR / "step6_static_variables.csv", index=False)
    rank = impact_ranking(df, reg, static_df)
    rank.to_csv(TAB_DIR / "step7_variable_impact_ranking.csv", index=False)

    # Step 8-10 y reporte
    label, why, *_ = classify(rank, reg)
    pd.DataFrame(dict(Hypothesis=hyp_tbl.Hypothesis, Outcome=list(reg[reg.kind == "hypothesis"].verdict) + list(static_df.verdict))).to_csv(TAB_DIR / "step8_hypothesis_outcomes.csv", index=False)
    write_variable_analysis_report(out_dir, df, reg, hyp_tbl, dash_tbl, models, regm, evid, rank, static_df, label, why, T, files)
    print(f"Classification: {label}\nEverything saved in {out_dir}")

if __name__ == "__main__":
    main()