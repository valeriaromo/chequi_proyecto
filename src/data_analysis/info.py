from __future__ import annotations
import calendar
import inspect
from datetime import date
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from helpers import (DATASET_SOURCE, DATASET_VERSION, LAB_OBJECTIVE, TEAM_MEMBERS,CORE, RELEVANCE, WET_MM, TAB_DIR, FACTS, REG,fmt, unit, skew_label, is_discrete, is_static, group_of,save_fig, reg, hbox, md_table, full_years, distinct_series, annual_rain_by_muni,)

# textos
TXT_NO_SPECIFIED = "Not specified (set it at the top of the script)"
TXT_REPORT_TITLE = "# Data Intelligence Lab - Steps 3 to 7"
TXT_SECTION_DOC = "## Dataset documentation"
TXT_SECTION_STEP3 = "## Step 3 - Measures of central tendency"
TXT_SECTION_STEP4 = "## Step 4 - Measures of dispersion"
TXT_SECTION_SANITY = "### Data sanity checks"
TXT_SECTION_STEP5 = "## Step 5 - Distribution visualisation"
TXT_SECTION_STEP6 = "## Step 6 - Comparative visualisation"
TXT_SECTION_STEP7 = "## Step 7 - Key findings (Finding -> Visualization -> Statistical evidence)"

# archivos de salida
OUT_REPORT = "analysis_report.md"
OUT_CT = "step3_central_tendency.csv"
OUT_DISP = "step4_dispersion.csv"
OUT_ANOM = "step5_anomaly_checks.csv"
OUT_FINDINGS = "step7_key_findings.csv"
OUT_CORR = "step6_spearman_correlation_core.csv"

# interpretación de los resultados
def interpret_central(c, mean, med, mode, sk, std, zero_share, static, n_unique):
    u = unit(c)
    if static:
        return (f"Static per municipality ({n_unique} distinct values): mean {fmt(mean)}{u} and median {fmt(med)}{u} describe the municipality panel, not temporal behaviour.")
    text = (f"Typical value is about {fmt(med)}{u} (median); mean {fmt(mean)}{u}, "f"mode {fmt(mode)}{u}. ")
    if zero_share >= 0.20:
        text += (f"{zero_share * 100:.0f}% of observations are exactly zero, so median and mode "f"describe the zero state while the mean is pulled up by non-zero values "f"({skew_label(sk)}, skewness {sk:.2f}). ")
    else:
        gap = (mean - med) / std if std > 0 else 0.0
        if abs(gap) < 0.1 and abs(sk) < 0.5:
            text += ("Mean and median nearly coincide: the distribution is close to symmetric and the mean is a reliable summary. ")
        else:
            side = "above" if gap > 0 else "below"
            text += (f"The mean sits {abs(gap):.2f} SD {side} the median; the distribution is "f"{skew_label(sk)} (skewness {sk:.2f}). ")
        if std > 0 and abs(mode - med) > 0.5 * std:
            text += ("The mode is far from the median, indicating concentration at another value (possible multimodality or truncation).")
    return text.strip()


def interpret_dispersion(c, mn, mx, q1, q3, std, cv, out_pct, static):
    u = unit(c)
    rng, iqr = mx - mn, q3 - q1
    if rng == 0:
        return "Constant variable (zero variance)."
    if static:
        return (f"Spread reflects only differences among municipalities (range {fmt(rng)}{u}); no temporal variation.")
    ratio = iqr / rng
    t = (f"Values span {fmt(mn)}-{fmt(mx)}{u}; the central 50% lie between {fmt(q1)} and "f"{fmt(q3)} (IQR {fmt(iqr)}, {ratio * 100:.0f}% of the range). ")
    if ratio < 0.15:
        t += "Observations are concentrated in a narrow core with long tails. "
    elif ratio > 0.40:
        t += "Observations are spread fairly evenly across the range. "
    if not np.isnan(cv):
        lab = ("very stable" if cv < 0.1 else "moderate variability" if cv < 0.3
               else "high variability" if cv < 1 else "extreme variability (SD exceeds the mean)")
        t += f"CV = {cv:.2f}: {lab}. "
    else:
        t += f"CV not applicable (interval/cyclic scale); SD = {fmt(std)}{u}. "
    if out_pct > 5:
        t += f"{out_pct:.1f}% of points fall outside Tukey fences: heavy tails."
    elif out_pct > 0:
        t += f"{out_pct:.1f}% of points fall outside Tukey fences."
    return t.strip()


# CALCULOS DE ESTADISTICA

def compute_mode(s, static):
    s = s.dropna()
    if static or is_discrete(s):
        m = s.mode()
    else:
        rng = s.max() - s.min()
        d = 2 if rng < 10 else 1 if rng < 1000 else 0
        m = s.round(d).mode()
    return float(m.iloc[0]) if len(m) else np.nan


def compute_central_tendency(df, qvars):
    rows = []
    for c in qvars:
        s = df[c].dropna().astype("float64")
        static = bool(is_static(df, c))
        mean, med, std = s.mean(), s.median(), s.std()
        mode = compute_mode(s, static)
        sk = float(s.skew()) if std > 0 else 0.0
        zero = float((s == 0).mean())
        rows.append(dict(
            Variable=c, Group=group_of(c), N=len(s), Mean=mean, Median=med,
            Mode=mode, Skewness=sk, Zero_share_pct=zero * 100,
            Interpretation=interpret_central(c, mean, med, mode, sk, std, zero, static, s.nunique()),
        ))
    return pd.DataFrame(rows)


def cv_value(c, s):
    interval = c.startswith("T2M")
    not_ratio = c in ("doy_sin", "doy_cos", "geom_centroid_lat", "geom_centroid_lon")
    if interval or not_ratio or abs(s.mean()) < 1e-9:
        return np.nan
    return float(s.std() / abs(s.mean()))


def compute_dispersion(df, qvars):
    rows = []
    for c in qvars:
        s = df[c].dropna().astype("float64")
        q1, q2, q3 = s.quantile([.25, .5, .75])
        iqr = q3 - q1
        out_pct = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean() * 100)
        cv = cv_value(c, s)
        mn, mx = s.min(), s.max()
        rows.append(dict(Variable=c, Group=group_of(c), Min=mn, Max=mx, Range=mx - mn,Variance=s.var(), Std_Dev=s.std(), Q1=q1, Median=q2, Q3=q3, IQR=iqr,CV=cv, Outliers_pct=out_pct,Interpretation=interpret_dispersion(c, mn, mx, q1, q3, s.std(), cv, out_pct,bool(is_static(df, c))),))
    return pd.DataFrame(rows)


def anomaly_checks(df, qvars):
    rows = []
    def add(name, mask, note):
        rows.append(dict(Check=name, Violations=int(mask.sum()),Share_pct=float(mask.mean() * 100), Note=note))

    if "RH2M" in df:
        add("RH2M outside [0,100]", (df.RH2M < 0) | (df.RH2M > 100), "physical bound")
    if "aod_qa_best_pct" in df:
        add("aod_qa_best_pct outside [0,100]",
            (df.aod_qa_best_pct < 0) | (df.aod_qa_best_pct > 100), "percentage bound")
    for c in ("precipitation_mm", "PRECTOTCORR", "WS2M", "vpd_kpa", "dias_sin_lluvia"):
        if c in df:
            add(f"{c} < 0", df[c] < 0, "non-negative quantity")
    if {"T2M_MAX", "T2M_MIN"} <= set(df.columns):
        add("T2M_MAX < T2M_MIN", df.T2M_MAX < df.T2M_MIN, "logical consistency")
        if "T2M" in df:
            add("T2M outside [T2M_MIN, T2M_MAX] (tol 0.05)",
                (df.T2M < df.T2M_MIN - .05) | (df.T2M > df.T2M_MAX + .05), "logical consistency")
    add("Any quantitative value <= -990 (fill-value sentinel)",
        (df[qvars] <= -990).any(axis=1), "NASA-style sentinel leaking through")
    return pd.DataFrame(rows)


# Monitoreo de skill score y estado del modelo
def kde_curve(x):
    try:
        if x.nunique() < 3 or x.std() == 0:
            return None, None
        xs = x.sample(10000, random_state=0) if len(x) > 10000 else x
        kde = stats.gaussian_kde(xs)
        grid = np.linspace(x.quantile(.001), x.quantile(.999), 400)
        return grid, kde(grid)
    except Exception:
        return None, None


def plot_distribution(df, c, ct_row, disp_row):
    s = df[c].dropna().astype("float64")
    zero = (s == 0).mean()
    u = unit(c)
    lo, hi = s.quantile([.001, .999])
    xs = s[(s >= lo) & (s <= hi)] if hi > lo else s
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(xs, bins=50, color="#4C78A8", alpha=.85)
    ax[0].axvline(s.mean(), color="#E45756", ls="--", label=f"Mean {fmt(s.mean())}")
    ax[0].axvline(s.median(), color="#54A24B", label=f"Median {fmt(s.median())}")
    if zero > .2:
        ax[0].set_yscale("log")
    ax[0].set_title("Histogram (0.1-99.9 percentile)")
    ax[0].set_xlabel(f"{c}{u}")
    ax[0].set_ylabel("Frequency" + (" (log)" if zero > .2 else ""))
    ax[0].legend()
    hbox(ax[1], s)
    ax[1].set_title("Box plot (all observations)")
    ax[1].set_xlabel(f"{c}{u}")
    ax[1].set_ylabel("All rows")
    ax[1].set_yticks([])
    dsrc = s[s > 0] if zero > .2 else s
    grid, dens = kde_curve(dsrc)
    peaks = 0
    if grid is not None:
        ax[2].fill_between(grid, dens, color="#F58518", alpha=.6,
                           label="Wet days only (>0)" if zero > .2 else "All observations")
        peaks = len(find_peaks(dens, prominence=.05 * dens.max())[0])
        ax[2].legend()
    ax[2].set_title("Density")
    ax[2].set_xlabel(f"{c}{u}")
    ax[2].set_ylabel("Density")
    fig.suptitle(f"Distribution of {c}", fontweight="bold")
    name = save_fig(fig, f"fig_step5_dist_{c}.png")
    shape = skew_label(ct_row.Skewness) + (f", possibly multimodal ({peaks} density peaks)" if peaks >= 2 else "")
    obs = (f"Shape: {shape}. Mean {fmt(ct_row.Mean)} vs median {fmt(ct_row.Median)}"
           f"{u}; {disp_row.Outliers_pct:.1f}% of points outside Tukey fences; "
           f"max {fmt(disp_row.Max)}{u}.")
    if zero > .2:
        obs += f" {zero * 100:.0f}% of observations are exactly zero (dominant concentration)."
    reg(5, name, f"Distribution of {c}",
        f"Histogram, box plot and density of {c} across all {len(s):,} non-missing observations.",
        obs, RELEVANCE.get(c, "Characterises the marginal distribution before modelling."))


def freq_table(series, label):
    vc = series.value_counts(dropna=False).sort_index()
    t = pd.DataFrame({label: vc.index.astype(str), "Count": vc.values,
                      "Percent": vc.values / vc.sum() * 100})
    t.to_csv(TAB_DIR / f"step5_frequency_{label}.csv", index=False)
    return t


def bar_with_labels(ax, labels, vals, color="#4C78A8"):
    b = ax.bar(labels, vals, color=color)
    tot = sum(vals)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height(), f"{v / tot * 100:.1f}%",
                ha="center", va="bottom", fontsize=7)


# visualización de la distribución de las variables cuantitativas
def fig_temperature(d):
    cols = [c for c in ["T2M_MAX", "T2M", "T2M_MIN"] if c in d]
    m = d.groupby(pd.Grouper(key="date", freq="MS"))[cols].mean()
    clim = d.groupby("month")[cols].mean()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for c in cols:
        ax.plot(m.index, m[c], label=c, lw=1.6)
    ax.set_title("Monthly mean temperature, average of all municipalities")
    ax.set_xlabel("Month")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    name = save_fig(fig, "fig_step6_timeseries_temperature.png")
    hot, cold = int(clim["T2M"].idxmax()), int(clim["T2M"].idxmin())
    amp = float(clim["T2M"].max() - clim["T2M"].min())
    FACTS["temp_season"] = (calendar.month_abbr[hot], float(clim.loc[hot, "T2M"]),
                            calendar.month_abbr[cold], float(clim.loc[cold, "T2M"]), amp)
    reg(6, name, "Monthly mean temperature over time",
        "T2M_MAX, T2M and T2M_MIN monthly means across the whole period. A line chart is the natural choice for ordered, continuous time.",
        f"Clear annual cycle: warmest month {calendar.month_abbr[hot]} ({clim.loc[hot, 'T2M']:.1f} °C), "
        f"coldest {calendar.month_abbr[cold]} ({clim.loc[cold, 'T2M']:.1f} °C), amplitude {amp:.1f} °C.",
        "Seasonality is the dominant temporal structure and motivates the day-of-year features and the climatology baseline.")


def fig_precip_time(d):
    tot = d.groupby(["municipio", pd.Grouper(key="date", freq="MS")])["precipitation_mm"].sum()
    monthly = tot.groupby(level=1).mean()
    clim = monthly.groupby(monthly.index.month).mean()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].bar(monthly.index, monthly.values, width=25, color="#4C78A8")
    ax[0].set_title("Monthly precipitation total (CHIRPS), average of all municipalities")
    ax[0].set_xlabel("Month")
    ax[0].set_ylabel("Precipitation (mm/month)")
    ax[1].bar([calendar.month_abbr[m] for m in clim.index], clim.values, color="#72B7B2")
    ax[1].set_title("Mean monthly precipitation by calendar month")
    ax[1].set_xlabel("Calendar month")
    ax[1].set_ylabel("Precipitation (mm/month)")
    name = save_fig(fig, "fig_step6_timeseries_precipitation.png")
    top = clim.nlargest(4)
    share = float(top.sum() / clim.sum() * 100)
    FACTS["rain_season"] = ([calendar.month_abbr[m] for m in top.index], share)
    reg(6, name, "Monthly precipitation over time and by calendar month",
        "Precipitation totals through time (top) and their seasonal profile (bottom). Bars are used because rainfall is an accumulated quantity.",
        f"The four wettest months ({', '.join(calendar.month_abbr[m] for m in top.index)}) contain "
        f"{share:.0f}% of the average annual rainfall; the wettest single month averages "
        f"{clim.max():.0f} mm.",
        "Strong concentration of rain in a few months explains the zero-inflation and the difficulty of the rain target in the dry season.")


def fig_geography(d):
    el = d.groupby("municipio")["elevacion_msnm"].first().sort_values()
    order = list(el.index)
    ann, basis = annual_rain_by_muni(d)
    ann = ann.reindex(order)
    data = [d.loc[d.municipio == m, "T2M_MIN"].values for m in order]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].boxplot(data, positions=range(1, len(order) + 1), showfliers=False)
    ax[0].set_xticks(range(1, len(order) + 1))
    ax[0].set_xticklabels([f"{m}\n{el[m]:.0f} m" for m in order], rotation=60, ha="right", fontsize=7)
    ax[0].set_title("T2M_MIN by municipality (ordered by elevation)")
    ax[0].set_xlabel("Municipality and elevation")
    ax[0].set_ylabel("T2M_MIN (°C)")
    ax[1].bar(range(len(order)), ann.values, color="#72B7B2")
    ax[1].set_xticks(range(len(order)))
    ax[1].set_xticklabels(order, rotation=60, ha="right", fontsize=7)
    ax[1].set_title(f"Annual precipitation by municipality ({basis})")
    ax[1].set_xlabel("Municipality (same order)")
    ax[1].set_ylabel("Precipitation (mm/year)")
    name = save_fig(fig, "fig_step6_geography_municipalities.png")
    med = d.groupby("municipio")["T2M_MIN"].median()
    n_t, n_m = distinct_series(d, "T2M")
    n_p, _ = distinct_series(d, "precipitation_mm")
    FACTS.update(tmin_med_range=(float(med.min()), float(med.max())),
                 rain_range=(float(ann.min()), float(ann.max()), ann.idxmin(), ann.idxmax()),
                 distinct_t2m=n_t, distinct_chirps=n_p, n_muni=n_m, rain_basis=basis)
    reg(6, name, "Geographic comparison across municipalities",
        "Distribution of T2M_MIN (box plots, because they compare spread and centre across groups) "
        "and annual rainfall (bars) for each municipality, ordered by elevation.",
        f"Median T2M_MIN differs by {med.max() - med.min():.1f} °C between municipalities; annual rainfall "
        f"ranges {ann.min():.0f}-{ann.max():.0f} mm ({ann.idxmin()} lowest, {ann.idxmax()} highest). "
        f"Only {n_t} of {n_m} municipalities have distinct NASA POWER T2M series and {n_p} have "
        f"distinct CHIRPS series.",
        "Shows whether municipality-level differences exist or are an artefact of the coarse source grids; "
        "this decides how much a global vs per-municipality model can exploit.")


def fig_elevation_scatter(d):
    el = d.groupby("municipio")["elevacion_msnm"].first()
    tm = d.groupby("municipio")["T2M"].mean()
    ann, basis = annual_rain_by_muni(d)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    res = {}
    for a, y, lab, key in ((ax[0], tm, "Mean T2M (°C)", "T"), (ax[1], ann, "Annual precipitation (mm)", "P")):
        x = el.reindex(y.index)
        a.scatter(x, y, color="#4C78A8", zorder=3)
        for m in y.index:
            a.annotate(m, (x[m], y[m]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        lr = stats.linregress(x, y)
        rho = stats.spearmanr(x, y)[0]
        xs = np.linspace(x.min(), x.max(), 20)
        a.plot(xs, lr.intercept + lr.slope * xs, color="#E45756",
               label=f"OLS fit: r = {lr.rvalue:.2f}, Spearman = {rho:.2f}, n = {len(x)}")
        a.set_xlabel("Elevation (m a.s.l.)")
        a.set_ylabel(lab)
        a.set_title(f"Elevation vs {lab}")
        a.legend(fontsize=7)
        res[key] = (float(lr.rvalue), float(rho), float(lr.pvalue))
    name = save_fig(fig, "fig_step6_scatter_elevation.png")
    FACTS["elev"] = res
    reg(6, name, "Elevation versus climate by municipality",
        "One point per municipality: elevation against mean temperature and annual rainfall "
        "(scatter plot to show the relationship and its strength).",
        f"Elevation-temperature r = {res['T'][0]:.2f} (Spearman {res['T'][1]:.2f}, p = {res['T'][2]:.2f}); "
        f"elevation-rainfall r = {res['P'][0]:.2f} (p = {res['P'][2]:.2f}). With only "
        f"{len(el)} points these are descriptive, not conclusive.",
        "Tests whether the static physical features carry a climate signal that can separate municipalities.")


def fig_nasa_vs_chirps(d):
    x, y = d["PRECTOTCORR"].astype(float), d["precipitation_mm"].astype(float)
    sp = float(stats.spearmanr(x, y)[0])
    pe = float(np.corrcoef(x, y)[0, 1])
    wx, wy = x >= WET_MM, y >= WET_MM
    cats = {"Both wet": (wx & wy).mean(), "Only CHIRPS wet": (~wx & wy).mean(),
            "Only NASA wet": (wx & ~wy).mean(), "Both dry": (~wx & ~wy).mean()}
    cap = max(x.quantile(.999), y.quantile(.999))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    hb = ax[0].hexbin(x.clip(upper=cap), y.clip(upper=cap), gridsize=60, bins="log", mincnt=1,
                      cmap="viridis")
    ax[0].plot([0, cap], [0, cap], color="#E45756",
               label=f"1:1 line (Spearman {sp:.2f}, Pearson {pe:.2f})")
    ax[0].set_xlabel("NASA POWER PRECTOTCORR (mm/day)")
    ax[0].set_ylabel("CHIRPS precipitation_mm (mm/day)")
    ax[0].set_title("Daily precipitation: NASA POWER vs CHIRPS")
    ax[0].legend(fontsize=7)
    fig.colorbar(hb, ax=ax[0], label="log10(count)")
    ax[1].bar(list(cats.keys()), [v * 100 for v in cats.values()],
              color=["#54A24B", "#F58518", "#B279A2", "#BAB0AC"])
    ax[1].set_title(f"Wet-day agreement (threshold {WET_MM:g} mm)")
    ax[1].set_xlabel("Classification")
    ax[1].set_ylabel("Share of observations (%)")
    plt.setp(ax[1].get_xticklabels(), rotation=20)
    name = save_fig(fig, "fig_step6_scatter_nasa_vs_chirps.png")
    agree = float((cats["Both wet"] + cats["Both dry"]) * 100)
    FACTS["nasa_chirps"] = (sp, pe, agree, cats)
    reg(6, name, "NASA POWER vs CHIRPS precipitation",
        "Day-by-day agreement of two independent rainfall products (hexbin scatter because the "
        "number of points hides overplotting; bars for wet/dry classification).",
        f"Spearman {sp:.2f}, Pearson {pe:.2f}; the products classify wet/dry the same way on "
        f"{agree:.1f}% of days, and disagree on {100 - agree:.1f}%.",
        "Quantifies the measurement uncertainty of the rain target and whether the two products are interchangeable.")


def fig_aod(d):
    miss = d.groupby("month")["aod_055_mean"].apply(lambda s: s.isna().mean() * 100)
    g = d.groupby("month")["aod_055_mean"]
    med = g.median()
    q1 = g.quantile(.25)
    q3 = g.quantile(.75)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].bar([calendar.month_abbr[m] for m in miss.index], miss.values, color="#E45756")
    ax[0].set_title("Share of rows with missing AOD by calendar month")
    ax[0].set_xlabel("Calendar month")
    ax[0].set_ylabel("Missing AOD (%)")
    xs = [calendar.month_abbr[m] for m in med.index]
    ax[1].plot(xs, med.values, marker="o", label="Median AOD")
    ax[1].fill_between(xs, q1.values, q3.values, alpha=.25, label="Interquartile range")
    ax[1].set_title("AOD level by calendar month (observed days)")
    ax[1].set_xlabel("Calendar month")
    ax[1].set_ylabel("aod_055_mean")
    ax[1].legend()
    name = save_fig(fig, "fig_step6_aod_monthly.png")
    mm, mn = int(miss.idxmax()), int(miss.idxmin())
    am = int(med.idxmax())
    FACTS["aod_month"] = (calendar.month_abbr[mm], float(miss.max()), calendar.month_abbr[mn],
                          float(miss.min()), calendar.month_abbr[am], float(med.max()))
    reg(6, name, "AOD availability and level by calendar month",
        "Seasonal pattern of missingness (bars) and of AOD values (median and IQR band).",
        f"Missingness peaks in {calendar.month_abbr[mm]} ({miss.max():.0f}%) and is lowest in "
        f"{calendar.month_abbr[mn]} ({miss.min():.0f}%); median AOD is highest in "
        f"{calendar.month_abbr[am]} ({med.max():.3f}).",
        "Missing AOD is not random through the year, so any AOD feature is informative only in part of the season.")


def fig_annual(d):
    full = full_years(d)
    if len(full) < 3:
        return
    g = d[d.yr.isin(full)]
    t = g.groupby("yr")["T2M"].mean()
    p = g.groupby(["municipio", "yr"])["precipitation_mm"].sum().groupby("yr").mean()
    lr = stats.linregress(t.index, t.values)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(t.index, t.values, marker="o", label="Annual mean T2M")
    ax[0].plot(t.index, lr.intercept + lr.slope * t.index, ls="--", color="#E45756",
               label=f"Linear trend {lr.slope:+.2f} °C/year (n = {len(t)} years)")
    ax[0].set_title("Annual mean temperature (full years only)")
    ax[0].set_xlabel("Year")
    ax[0].set_ylabel("T2M (°C)")
    ax[0].legend(fontsize=7)
    ax[1].bar(p.index.astype(str), p.values, color="#4C78A8")
    ax[1].set_title("Annual precipitation (full years only)")
    ax[1].set_xlabel("Year")
    ax[1].set_ylabel("Precipitation (mm/year)")
    name = save_fig(fig, "fig_step6_annual_comparison.png")
    FACTS["annual"] = (full, float(t.idxmax()), float(t.max()), int(p.idxmax()), float(p.max()),
                       int(p.idxmin()), float(p.min()), float(lr.slope))
    reg(6, name, "Year-to-year comparison",
        "Annual temperature and rainfall for complete years (the partial final year is excluded to "
        "avoid seasonal bias).",
        f"Warmest year {int(t.idxmax())} ({t.max():.1f} °C); wettest {int(p.idxmax())} "
        f"({p.max():.0f} mm), driest {int(p.idxmin())} ({p.min():.0f} mm); temperature trend "
        f"{lr.slope:+.2f} °C/year over only {len(t)} years.",
        "Shows inter-annual variability, which is what the walk-forward folds must absorb.")


def fig_correlation(d):
    cols = [c for c in CORE if c in d]
    cm = d[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(cm.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=75, ha="right", fontsize=7)
    ax.set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Spearman correlation of core variables")
    name = save_fig(fig, "fig_step6_correlation_heatmap.png")
    pairs = [(cols[i], cols[j], float(cm.values[i, j])) for i in range(len(cols))
             for j in range(i + 1, len(cols)) if abs(cm.values[i, j]) >= .9]
    pairs.sort(key=lambda p: -abs(p[2]))
    FACTS["corr_pairs"] = pairs
    cm.to_csv(TAB_DIR / OUT_CORR)
    top = "; ".join(f"{a}-{b} ({r:.2f})" for a, b, r in pairs[:4]) or "none"
    reg(6, name, "Correlation among core variables",
        "Pairwise rank correlation of the core variables (heatmap; Spearman because several "
        "variables are skewed or zero-inflated).",
        f"{len(pairs)} variable pairs have |rho| >= 0.90. Strongest: {top}.",
        "Identifies redundant predictors and highlights which relationships deserve modelling attention.")


# descripcion de hallazgos y anomalías
def build_findings(df, ct, disp, anomalies):
    cti, di = ct.set_index("Variable"), disp.set_index("Variable")
    F = []

    def add(priority, finding, viz, evidence):
        F.append(dict(Priority=priority, Finding=finding, Visualization=viz,Statistical_Evidence=evidence))

    if "precipitation_mm" in cti.index:
        r, q = cti.loc["precipitation_mm"], di.loc["precipitation_mm"]
        add("High",
            f"Daily precipitation is zero-inflated and strongly right-skewed: {r.Zero_share_pct:.0f}% of "
            f"days are exactly zero, so mean ({fmt(r.Mean)} mm) and median ({fmt(r.Median)} mm) tell "
            f"different stories.",
            "fig_step5_dist_precipitation_mm.png; fig_step6_timeseries_precipitation.png",
            f"Step 3 table (precipitation_mm): mean {fmt(r.Mean)}, median {fmt(r.Median)}, mode {fmt(r.Mode)}, "
            f"skewness {r.Skewness:.2f}. Step 4 table: max {fmt(q.Max)} mm, std {fmt(q.Std_Dev)}, "
            f"{q.Outliers_pct:.1f}% beyond Tukey fences.")
    if "distinct_t2m" in FACTS:
        n, m, p = FACTS["distinct_t2m"], FACTS["n_muni"], FACTS["distinct_chirps"]
        txt = (f"Only {n} of {m} municipalities have distinct NASA POWER temperature series and {p} have "
               f"distinct CHIRPS rainfall series")
        txt += (": municipal differences in NASA variables partly reflect shared grid cells, not local climate."
                if n < m else ": every municipality has its own series.")
        add("High", txt, "fig_step6_geography_municipalities.png",
            f"Series de-duplication of T2M (round to 3 decimals) -> {n} distinct of {m}; "
            f"precipitation_mm -> {p} distinct of {m}.")
    if "aod_missing_pct" in FACTS and "aod_month" in FACTS:
        a = FACTS["aod_month"]
        add("High",
            f"AOD is missing in {FACTS['aod_missing_pct']:.1f}% of rows and the gaps are seasonal "
            f"(highest in {a[0]}: {a[1]:.0f}%, lowest in {a[2]}: {a[3]:.0f}%).",
            "fig_step6_aod_monthly.png; fig_step5_flags_and_aod_availability.png",
            f"{FACTS['aod_missing_n']:,} missing rows in aod_055_mean; frequency tables "
            f"step5_frequency_aod_availability.csv and step5_frequency_qc_aod_pocos_pixeles.csv.")
    if "temp_season" in FACTS and "rain_season" in FACTS:
        t, rs = FACTS["temp_season"], FACTS["rain_season"]
        add("High",
            f"Strong seasonality: mean temperature swings {t[4]:.1f} °C between {t[2]} and {t[0]}, and the four "
            f"wettest months ({', '.join(rs[0])}) hold {rs[1]:.0f}% of annual rainfall.",
            "fig_step6_timeseries_temperature.png; fig_step6_timeseries_precipitation.png",
            f"Calendar-month climatology: T2M {t[3]:.1f} °C ({t[2]}) to {t[1]:.1f} °C ({t[0]}); "
            f"rain share computed from monthly totals averaged over municipalities.")
    if "elev" in FACTS:
        e, tr, rr = FACTS["elev"], FACTS["tmin_med_range"], FACTS["rain_range"]
        add("Medium",
            f"Geographic contrast exists but rests on few points: elevation-temperature r = {e['T'][0]:.2f}, "
            f"elevation-rainfall r = {e['P'][0]:.2f} (n = {FACTS['n_muni']} municipalities); median T2M_MIN "
            f"spans {tr[0]:.1f} to {tr[1]:.1f} °C and annual rainfall {rr[0]:.0f} ({rr[2]}) to {rr[1]:.0f} ({rr[3]}) mm.",
            "fig_step6_scatter_elevation.png; fig_step6_geography_municipalities.png",
            f"Pearson/Spearman on municipality means (p = {e['T'][2]:.2f} for temperature, "
            f"{e['P'][2]:.2f} for rainfall); rainfall basis: {FACTS['rain_basis']}.")
    if "nasa_chirps" in FACTS:
        sp, pe, ag, _ = FACTS["nasa_chirps"]
        add("Medium",
            f"Two rainfall products disagree materially at daily scale: Spearman {sp:.2f}, and they classify wet vs dry "
            f"identically on {ag:.1f}% of days.",
            "fig_step6_scatter_nasa_vs_chirps.png",
            f"Pearson {pe:.2f}, Spearman {sp:.2f}, wet threshold {WET_MM:g} mm; agreement categories in the figure.")
    core = [c for c in CORE if c in di.index and not np.isnan(di.loc[c, "CV"])]
    if core:
        cvs = di.loc[core, "CV"].sort_values()
        hi, lo = cvs.index[-3:][::-1], cvs.index[:3]
        add("Medium",
            f"Highest relative variability: {', '.join(f'{c} (CV {cvs[c]:.2f})' for c in hi)}; most stable: "
            f"{', '.join(f'{c} (CV {cvs[c]:.2f})' for c in lo)}.",
            "; ".join(f"fig_step5_dist_{c}.png" for c in list(hi[:2]) + list(lo[:1])),
            "Step 4 table, CV column (ratio-scale core variables only; temperatures excluded because Celsius has no true zero).")
    if "Outliers_pct" in di.columns:
        oc = di.loc[[c for c in CORE if c in di.index], "Outliers_pct"].sort_values(ascending=False)[:3]
        viol = int(anomalies["Violations"].sum())
        add("Medium",
            f"Heaviest tails: {', '.join(f'{c} ({v:.1f}% outside Tukey fences)' for c, v in oc.items())}; "
            f"physical-consistency checks found {viol} violation(s).",
            "; ".join(f"fig_step5_dist_{c}.png" for c in oc.index[:2]),
            "Step 4 table, Outliers_pct column; table step5_anomaly_checks.csv.")
    if "annual" in FACTS:
        a = FACTS["annual"]
        add("Medium",
            f"Year-to-year variability is notable: wettest full year {a[3]} ({a[4]:.0f} mm) vs driest {a[5]} ({a[6]:.0f} mm); "
            f"temperature trend {a[7]:+.2f} °C/year over only {len(a[0])} full years.",
            "fig_step6_annual_comparison.png",
            f"Annual aggregates over full years {a[0][0]}-{a[0][-1]}; trend by OLS (descriptive, small n).")
    if "corr_pairs" in FACTS:
        pr = FACTS["corr_pairs"]
        add("Medium",
            f"{len(pr)} pairs of core variables are nearly redundant (|rho| >= 0.90)"
            + (f", e.g. {pr[0][0]} and {pr[0][1]} (rho {pr[0][2]:.2f})." if pr else "."),
            "fig_step6_correlation_heatmap.png", "table step6_spearman_correlation_core.csv.")
    mc = FACTS.get("muni_counts")
    tot = int(df.isna().sum().sum())
    add("Low",
        f"The panel is {'balanced' if mc and mc[0] == mc[1] else 'unbalanced'} "
        f"({mc[0]:,}-{mc[1]:,} rows per municipality) and {tot:,} cells "
        f"({tot / df.size * 100:.2f}%) remain missing, concentrated in AOD and in the start-of-series lags/windows.",
        "fig_step5_municipio_frequency.png",
        "Frequency table step5_frequency_municipio.csv; df.isna().sum().")
    out = pd.DataFrame(F)
    out.insert(0, "ID", [f"F{i + 1}" for i in range(len(out))])
    out.to_csv(TAB_DIR / OUT_FINDINGS, index=False)
    return out


# ===========================================================================
# REPORT BUILDER
# ===========================================================================
def write_report(out_dir, df, qvars, ct, disp, anomalies, findings, data_path):
    nm = lambda v: v if v else TXT_NO_SPECIFIED
    L = [TXT_REPORT_TITLE, "", TXT_SECTION_DOC, "",
         f"- **Dataset source:** {DATASET_SOURCE}",
         f"- **File analysed:** `{data_path}`",
         f"- **Dataset version:** {nm(DATASET_VERSION)}",
         f"- **Date of analysis:** {date.today().isoformat()}",
         f"- **Objective of the Data Intelligence Lab:** {nm(LAB_OBJECTIVE)}",
         f"- **Team members:** {nm(TEAM_MEMBERS)}", "",
         f"Rows: {len(df):,}. Columns: {df.shape[1]}. Quantitative variables analysed: {len(qvars)} "
         f"(flags, calendar and cyclic day-of-year variables are treated separately).", "",
         TXT_SECTION_STEP3, "", "**Calculation** (code used):", "", "```python"]
    for f in (compute_mode, compute_central_tendency, interpret_central):
        L.append(inspect.getsource(f))
    L += ["```", "",
          "**Result and interpretation** (full table with skewness: `tables/step3_central_tendency.csv`):", "",
          md_table(ct[["Variable", "Mean", "Median", "Mode", "Interpretation"]]), "",
          f"Summary: {int((ct.Skewness.abs() < .5).sum())} variables are approximately symmetric, "
          f"{int(((ct.Skewness.abs() >= .5) & (ct.Skewness.abs() < 1)).sum())} moderately skewed and "
          f"{int((ct.Skewness.abs() >= 1).sum())} strongly skewed. Mode on continuous variables is computed on rounded values.", "",
          TXT_SECTION_STEP4, "", "**Calculation** (code used):", "", "```python"]
    for f in (cv_value, compute_dispersion, interpret_dispersion):
        L.append(inspect.getsource(f))
    L += ["```", "",
          "**Result and interpretation** (full table with variance and quartiles: `tables/step4_dispersion.csv`):", "",
          md_table(disp[["Variable", "Min", "Max", "Range", "Std_Dev", "IQR", "Interpretation"]]), "",
          TXT_SECTION_SANITY, "", md_table(anomalies), "", TXT_SECTION_STEP5, ""]
    for r in [x for x in REG if x["step"] == 5]:
        L += [f"### {r['title']}", "", f"![{r['title']}](figures/{r['file']})", "",
              f"**Description:** {r['compared']}", "", f"**Key observations:** {r['obs']}", "",
              f"**Relevance:** {r['relevance']}", ""]
    L += [TXT_SECTION_STEP6, ""]
    for r in [x for x in REG if x["step"] == 6]:
        L += [f"### {r['title']}", "", f"![{r['title']}](figures/{r['file']})", "",
              f"**What is compared and why this chart:** {r['compared']}", "",
              f"**Main observations:** {r['obs']}", "", f"**Relevance to the project:** {r['relevance']}", ""]
    L += [TXT_SECTION_STEP7, "", md_table(findings), ""]
    (out_dir / OUT_REPORT).write_text("\n".join(L), encoding="utf-8")