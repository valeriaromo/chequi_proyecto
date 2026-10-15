# correr python3 data_analysis.py [--data PATH] [--out DIR] si cambian las rutas default
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from helpers import (HERE, BASE_DIR, CORE, FIG_DIR, TAB_DIR, quant_vars, save_fig, reg,)
from info import (compute_central_tendency, compute_dispersion, anomaly_checks,plot_distribution, freq_table, bar_with_labels,fig_temperature, fig_precip_time, fig_geography, fig_elevation_scatter,fig_nasa_vs_chirps, fig_aod, fig_annual, fig_correlation,build_findings, write_report,OUT_CT, OUT_DISP, OUT_ANOM,)

# distribuciones cualitativas y cuantitativas, anomalías, hallazgos y reportes
def step5_qualitative(df):
    import calendar
    import matplotlib.pyplot as plt
    import numpy as np
    from helpers import FLAGS, FACTS

    # municipality
    t = freq_table(df["municipio"], "municipio")
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_with_labels(ax, t["municipio"], t["Count"])
    ax.set_title("Observations per municipality")
    ax.set_xlabel("Municipality")
    ax.set_ylabel("Number of observations")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    name = save_fig(fig, "fig_step5_municipio_frequency.png")
    reg(5, name, "Observations per municipality",
        "Row counts for each of the municipalities.",
        f"Counts range {t.Count.min():,}-{t.Count.max():,}; "
        f"{'the panel is perfectly balanced' if t.Count.nunique() == 1 else 'the panel is unbalanced'}. "
        "A pie chart was not used because equal slices add no information.",
        "Confirms every municipality has equal weight in any pooled statistic or model.")
    FACTS["muni_counts"] = (int(t.Count.min()), int(t.Count.max()), len(t))

    # por mes
    tm = freq_table(df["month"], "month")
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_with_labels(ax, [calendar.month_abbr[int(m)] for m in tm["month"]], tm["Count"], "#72B7B2")
    ax.set_title("Observations per calendar month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of observations")
    name = save_fig(fig, "fig_step5_month_frequency.png")
    reg(5, name, "Observations per calendar month", "Row counts for each calendar month.",
        f"Fewest observations in {calendar.month_abbr[int(tm.loc[tm.Count.idxmin(), 'month'])]} "
        f"({tm.Count.min():,}), most in {calendar.month_abbr[int(tm.loc[tm.Count.idxmax(), 'month'])]} "
        f"({tm.Count.max():,}); the imbalance comes from the last year being incomplete.",
        "Months covered by fewer years carry less evidence in seasonal comparisons.")

    # flags y disponibilidad de AOD
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, f in enumerate(FLAGS):
        if f in df:
            t = freq_table(df[f], f)
            bar_with_labels(ax[i], [f"{v}" for v in t[f]], t["Count"], ["#B279A2", "#9D755D"][i])
            ax[i].set_title(f)
            ax[i].set_xlabel("Flag value (0 = not flagged, 1 = flagged)")
            ax[i].set_ylabel("Number of observations")
    avail = df["aod_055_mean"].notna().map({True: "AOD observed", False: "AOD missing"})
    ta = freq_table(avail, "aod_availability")
    ax[2].pie(ta["Count"], labels=ta["aod_availability"], autopct="%1.1f%%",
              colors=["#4C78A8", "#E45756"][:len(ta)], startangle=90)
    ax[2].set_title("AOD availability")
    ax[2].grid(False)
    fig.suptitle("Binary quality flags and AOD availability", fontweight="bold")
    name = save_fig(fig, "fig_step5_flags_and_aod_availability.png")
    miss = float(df["aod_055_mean"].isna().mean() * 100)
    FACTS["aod_missing_pct"] = miss
    FACTS["aod_missing_n"] = int(df["aod_055_mean"].isna().sum())
    pocos = float(df["qc_aod_pocos_pixeles"].mean() * 100) if "qc_aod_pocos_pixeles" in df else np.nan
    reg(5, name, "Binary quality flags and AOD availability",
        "Proportion of rows per flag value, and of rows with vs without an AOD retrieval "
        "(pie chart used because the two shares sum to a whole).",
        f"AOD is missing in {miss:.1f}% of rows; {pocos:.1f}% of rows carry the few-pixels flag; "
        f"qc_aod_imputed is {'constant (no imputation)' if df.qc_aod_imputed.nunique() == 1 else 'not constant'}.",
        "Defines how much of the AOD signal is usable and confirms no gap-filling.")


def step5(df, qvars, ct, disp):
    cti, di = ct.set_index("Variable"), disp.set_index("Variable")
    for c in [v for v in CORE if v in df]:
        r1, r2 = cti.loc[c], di.loc[c]
        r1.name = c
        plot_distribution(df, c, r1, r2)
    step5_qualitative(df)


# orquestador de los pasos 3-7 de análisis de datos
def step6(d):
    fig_temperature(d)
    fig_precip_time(d)
    fig_geography(d)
    fig_elevation_scatter(d)
    fig_nasa_vs_chirps(d)
    fig_aod(d)
    fig_annual(d)
    fig_correlation(d)


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
    ap.add_argument("--data", default=None, help="path to master_table.parquet")
    ap.add_argument("--out", default=None, help="output folder (default: src/data_analysis/output)")
    a = ap.parse_args()
    data_path = Path(a.data) if a.data else default_master_path()
    out_dir = Path(a.out) if a.out else HERE / "output"
    FIG_DIR, TAB_DIR = out_dir / "figures", out_dir / "tables"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df["municipio"] = df["municipio"].astype(str)
    df = df.sort_values(["municipio", "date"]).reset_index(drop=True)
    df["yr"] = df["date"].dt.year
    qvars = quant_vars(df.drop(columns="yr"))
    print(f"Loaded {data_path} -> {df.shape[0]:,} rows, {df.shape[1] - 1} columns, {len(qvars)} quantitative")

    ct = compute_central_tendency(df, qvars)
    ct.to_csv(TAB_DIR / OUT_CT, index=False)
    disp = compute_dispersion(df, qvars)
    disp.to_csv(TAB_DIR / OUT_DISP, index=False)
    anomalies = anomaly_checks(df, qvars)
    anomalies.to_csv(TAB_DIR / OUT_ANOM, index=False)
    print("Steps 3-4 done")

    step5(df, qvars, ct, disp)
    print("Step 5 done")
    step6(df)
    print("Step 6 done")
    findings = build_findings(df, ct, disp, anomalies)
    write_report(out_dir, df, qvars, ct, disp, anomalies, findings, data_path)
    print(f"Step 7 done. Everything saved in {out_dir}")


if __name__ == "__main__":
    main()