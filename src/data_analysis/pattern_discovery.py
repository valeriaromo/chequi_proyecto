from __future__ import annotations
import argparse
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pca_analysis as pca
from helpers import DATASET_SOURCE, DATASET_VERSION, TEAM_MEMBERS, md_table
from variable_analysis import HERE, build_anomalies, default_master_path, distinct_series, load_data, md_src

OBJECTIVE = ("Discover hidden multivariate structure in the daily climate variables of the 11 municipalities after removing the annual cycle, and decide readiness for segmentation.")
RANK_CSV = HERE / "output" / "variable_analysis" / "tables" / "step7_variable_impact_ranking.csv"
OUT_DIR = HERE / "output" / "data_analysis"

# ORQUESTRADOR (o con --profile en la terminal)

#   "base8"  las 8 variables originales (con AOD) seguún el variable analysis
#   "sin_aod" 9 variables: sin AOD, con PS y precipitation_mm, log1p en lluvia
#   "ambos"   -> corre las dos y guarda cada una en su carpeta
PROFILE = "sin_aod"

# criterios de aceptación
REDUCTION_MAX = 0.60
STABILITY_MIN = 0.90
SKEW_MIN, AUTOCORR_MIN = 0.5, 0.3
GROUP_ETA_MIN = 5.0

DEFAULT_WHY = ("Removes the annual cycle so PCA describes day-to-day structure; z-scoring puts °C, %, m/s, mm and AOD on one scale (PCA on the correlation matrix).")


def why(v: str, profile: str) -> str:
    log = v in pca.LOG1P[profile]
    if v == pca.AOD:
        return ("Missing days are seasonal (clear-sky retrievals): dropping rows would bias the sample, so the gap is set to the mean "
                "(= climatology); Step 4 checks the effect with a PCA without AOD.")
    if v in ("PRECTOTCORR", "precipitation_mm"):
        return ("Zero-inflated: log1p is applied before the anomaly so rare heavy-rain days do not dominate; PCA is linear, "
                "so read its loading with caution." if log else
                "Zero-inflated: its anomalies are skewed and PCA is linear, so read its loading with caution.")
    return DEFAULT_WHY


def analyse(df: pd.DataFrame, anomalies: pd.DataFrame, profile: str) -> SimpleNamespace:
    X = anomalies[pca.PCA_VARS]
    Z = pca.standardise(X)
    full = pca.fit_pca(Z)
    aod = pca.has_aod()
    no_aod = pca.fit_pca(Z.drop(columns=pca.AOD)) if aod else None
    raw = pca.fit_pca(pca.standardise(df[pca.PCA_VARS]))
    meta = df[["municipio", "date", "yr"]].assign(month=df.date.dt.month)
    if aod:
        meta["aod_obs"] = X[pca.AOD].notna()
    k80 = pca.components_needed(full)
    pcs = [f"PC{i + 1}" for i in range(k80)]
    d2, threshold = pca.unusual_days(full, n=k80)
    return SimpleNamespace(
        profile=profile, df=df, X=X, full=full, no_aod=no_aod, raw=raw, meta=meta, d2=d2, threshold=threshold,
        flag=d2 > threshold, k80=k80, pcs=pcs,
        kaiser=int((full.eigenvalues > 1).sum()),
        similarity=pca.loading_similarity(full, no_aod, n=k80) if aod else None,
        structure=pca.structure_table(full.scores, meta, n=k80),
        autocorr=pca.lag1_autocorr(full.scores, meta.municipio, n=k80),
        shape=pca.shape_stats(full.scores, n=k80),
        interp={pc: pca.interpret(full.loadings[pc]) for pc in pcs},
        distinct=distinct_series(df.assign(PC1=full.scores.PC1.values), "PC1")[0])


def classify(r: SimpleNamespace) -> tuple[str, str]:
    checks = {
        f"{r.k80} components retain 80% of {len(pca.PCA_VARS)} variables (needed <= {REDUCTION_MAX:.0%})":
            r.k80 <= REDUCTION_MAX * len(pca.PCA_VARS),
        f"scores show skew or persistence (max |skew| {r.shape.Skewness.abs().max():.2f}, max lag-1 autocorrelation "
        f"{r.autocorr.max():.2f}; needed >= {SKEW_MIN} or >= {AUTOCORR_MIN})":
            r.shape.Skewness.abs().max() >= SKEW_MIN or r.autocorr.max() >= AUTOCORR_MIN,
    }
    if r.similarity is not None:
        checks[f"PC1-PC{r.k80} loadings stay stable without AOD (min |cosine| {r.similarity.min():.2f}, needed >= {STABILITY_MIN})"] = \
            r.similarity.min() >= STABILITY_MIN
    passed = sum(checks.values())
    label = ("READY FOR SEGMENTATION" if passed == len(checks) else
             "CONDITIONALLY READY" if passed else "ADDITIONAL DATA REQUIRED")
    why_txt = f"{passed} of {len(checks)} checks passed: " + "; ".join(f"{k} -> {'pass' if v else 'fail'}" for k, v in checks.items()) + "."
    return label, why_txt


def step1(r):
    rank = pd.read_csv(RANK_CSV).set_index("Variable")
    rows = [dict(Feature=v, Description=pca.FEATURES[v][0], Extraction_Method=pca.FEATURES[v][1],
                 Analytical_Relevance=f"{rank.Impact_Level[v]} (rank {rank.Rank[v]} of {len(rank)}); {rank.Statistical_Evidence[v]}")
            for v in pca.PCA_VARS]
    left_out = [v for v in ("PS", "precipitation_mm", pca.AOD) if v not in pca.PCA_VARS]
    excluded = ("`t2m_rango` and `vpd_kpa` (derived from T2M_MAX, T2M_MIN and RH2M, they would dominate by construction), "
                + "".join(f"`{v}`, " for v in left_out)
                + "`rain_30d` and `dias_sin_lluvia` (Low impact, zero-inflated), `aod_qa_best_pct` and "
                "`qc_aod_imputed` (constant), static variables (11 points) and the targets. "
                + ("AOD is excluded: 20% missing with seasonal gaps, no consistent gain in XGBoost and no signal against PM2.5."
                   if pca.AOD in left_out else
                   "AOD is kept in this profile for comparison with the profile without it."))
    return ["## Step 1 - Analytical variable selection", "",
            "Selected: variables with moderate or strong deseasonalised relationships, High or Medium impact in the Variable "
            "Impact Analysis, and relevance to the lab objective.", "",
            "Excluded: " + excluded, "",
            md_table(pd.DataFrame(rows)), ""]


def step2(r):
    n, miss = len(r.X), r.X.isna().sum()
    rows = [dict(Variable=v, Standardized="Yes",
                 Transformation_Applied=("log1p, then " if v in pca.LOG1P[r.profile] else "")
                 + "15-day smoothed day-of-year anomaly per municipality, then z-score"
                 + (f"; {miss[v]:,} NaN ({miss[v] / n * 100:.1f}%) set to 0 after z-scoring" if miss[v] else ""),
                 Justification=why(v, r.profile)) for v in pca.PCA_VARS]
    return ["## Step 2 - Multivariate data preparation", "",
            f"Rows used: {n:,} (the full master table; no row is dropped). Missing values: only "
            f"{', '.join(miss[miss > 0].index) or 'none'} has gaps. Numerical representation: all variables are continuous floats, "
            "no encoding is needed; `municipio` and `date` are labels, not inputs. Standardisation and scaling: required, because "
            "the units differ.", "",
            "**Code used:**", "", md_src(pca.standardise, pca.fit_pca), "",
            md_table(pd.DataFrame(rows)), ""]


def step3(r):
    res, raw = r.full, r.raw
    summary = pd.DataFrame({"Component": res.loadings.columns,
                            "Explained_Variance_%": (res.ratio * 100).round(1),
                            "Cumulative_Variance_%": (res.cumulative * 100).round(1)})
    p = len(pca.PCA_VARS)
    return ["## Step 3 - Principal Component Analysis", "",
            f"{p} components were generated (one per variable). {r.k80} reach {pca.VARIANCE_TARGET:.0%} cumulative variance and "
            f"{r.kaiser} have eigenvalue > 1 (Kaiser). PC1-PC{r.k80} retain {res.cumulative[r.k80 - 1] * 100:.1f}% of the "
            f"information, so dimensionality can be reduced from {p} to {r.k80} with {res.cumulative[r.k80 - 1] * 100:.1f}% retained. "
            f"Reference: on raw, non-deseasonalised values PC1 alone would explain {raw.ratio[0] * 100:.1f}% "
            f"(PC1-PC{r.k80}: {raw.cumulative[r.k80 - 1] * 100:.1f}%), mostly the annual cycle.", "",
            md_table(summary), ""], summary


def step4(r):
    res = r.full
    rows = [dict(Component=pc, Dominant_Variables=r.interp[pc][0], Interpretation=r.interp[pc][1]) for pc in r.pcs]
    loads = res.loadings[r.pcs].round(2).reset_index(names="Variable")
    out = ["## Step 4 - Component interpretation", "",
           "Loadings are the correlation between each original variable and the component; the sign of a component is arbitrary, "
           "only the contrast between variables matters. Interpretations are generated from the dominant loadings "
           f"(|loading| >= {pca.LOADING_MIN}) and should be confirmed physically. Components are analysed up to PC{r.k80} "
           f"(the number needed to retain {pca.VARIANCE_TARGET:.0%} of the variance).", "",
           "**Code used:**", "", md_src(pca.interpret), "",
           md_table(loads), "", md_table(pd.DataFrame(rows)), ""]
    sens = None
    if r.no_aod is not None:
        sens = pd.DataFrame({"Component": r.pcs,
                             "Variance_with_AOD_%": (res.ratio[:r.k80] * 100).round(1),
                             "Variance_without_AOD_%": (r.no_aod.ratio[:r.k80] * 100).round(1),
                             "Loading_similarity": r.similarity.round(2)})
        aod = res.loadings.loc[pca.AOD, r.pcs]
        out += [f"**AOD sensitivity.** AOD loads most on {aod.abs().idxmax()} ({aod[aod.abs().idxmax()]:+.2f}). "
                "Comparison with a PCA that excludes AOD (Loading_similarity = |cosine| of the loadings on the common variables):", "",
                md_table(sens), ""]
    return out, rows, loads, sens


def step5(r, files):
    res, st = r.full, r.structure
    rows = [
        dict(Visualization="Explained variance", What_is_visualized="Variance per component and cumulative.",
             Main_observations=f"PC1-PC{r.k80} hold {res.cumulative[r.k80 - 1] * 100:.1f}%; {r.k80} components reach 80%.",
             Analytical_relevance="Sets how many dimensions the next stages need."),
        dict(Visualization="Component loadings", What_is_visualized=f"Contribution of each variable to PC1-PC{r.k80}.",
             Main_observations="; ".join(f"{pc}: {r.interp[pc][1].lower()}" for pc in r.pcs) + ".",
             Analytical_relevance="Shows which variables form each latent dimension."),
        dict(Visualization="Two-component projections",
             What_is_visualized=f"Daily observations on all PC pairs among PC1-PC{r.k80}.",
             Main_observations=f"{int(r.flag.sum()):,} days ({r.flag.mean() * 100:.2f}%) fall beyond the {pca.UNUSUAL_P:.1%} chi-square "
                               f"limit (expected 0.1% if scores were Gaussian); skewness {', '.join(f'{v:+.2f}' for v in r.shape.Skewness)}.",
             Analytical_relevance="Reveals the shape of the cloud, tails and candidate groups."),
        dict(Visualization="PC1 vs PC2 by month", What_is_visualized="Same projection coloured by calendar month.",
             Main_observations=f"Month explains {st.loc['Month', 'PC1']:.1f}% of PC1 and {st.loc['Month', 'PC2']:.1f}% of PC2 variance.",
             Analytical_relevance="Checks that deseasonalising removed the annual cycle."),
    ]
    figs = ["### " + t + "\n\n![" + t + f"](figures/{f})\n" for t, f in files.items()]
    return ["## Step 5 - Visual analytics", "", "\n".join(figs), "### Pattern Discovery Dashboard", "",
            md_table(pd.DataFrame(rows)), ""], rows


def step6(r):
    st, shape, big = r.structure, r.shape, r.structure.stack().idxmax()
    top = r.d2.groupby(r.meta.date).max().nlargest(3)
    months = r.meta.month[r.flag].value_counts().head(3)
    rows = [
        ("Emerging structures", f"Skewness / excess kurtosis of PC1-PC{r.k80}: " +
         "; ".join(f"{pc} {shape.Skewness[pc]:+.2f} / {shape.Excess_kurtosis[pc]:+.2f}" for pc in r.pcs) + "."),
        ("Potential groups", f"Largest grouping effect: {big[0]} explains {st.loc[big]:.1f}% of {big[1]}; "
         + ("at least one grouping separates the data." if st.to_numpy().max() >= GROUP_ETA_MIN
            else f"no grouping reaches {GROUP_ETA_MIN:.0f}%, so no discrete groups are visible in PC1-PC{r.k80}.")),
        ("Similar observations", f"The 11 municipalities collapse to {r.distinct} distinct score series: municipalities on the same "
         "NASA grid cell are interchangeable."),
        ("Separation zones", f"{', '.join(st.index).lower()} explain "
         f"{st.to_numpy().min():.1f}-{st.to_numpy().max():.1f}% of any PC's variance."),
        ("Unusual observations", f"{int(r.flag.sum()):,} rows ({r.flag.mean() * 100:.2f}%) beyond the chi-square limit; most extreme "
         f"dates {', '.join(d.strftime('%Y-%m-%d') for d in top.index)}; most frequent months {', '.join(map(str, months.index))}."),
        ("Hidden tendencies", "Day-to-day autocorrelation of the scores: " +
         "; ".join(f"{pc} {r.autocorr[pc]:.2f}" for pc in r.pcs) + " (persistent anomalies suggest weather regimes)."),
        ("Dominant dimensions", f"{r.k80} components explain {r.full.cumulative[r.k80 - 1] * 100:.1f}%; PC1 alone "
         f"{r.full.ratio[0] * 100:.1f}%: {r.interp['PC1'][1].lower()}."),
    ]
    return ["## Step 6 - Hidden pattern identification", "",
            md_table(pd.DataFrame(rows, columns=["Aspect", "Finding"])), "",
            f"Share of each component's variance (PC1-PC{r.k80}) explained by a grouping (%):", "",
            md_table(st.round(1).reset_index(names="Grouping")), ""], rows


def step7(r):
    res, p = r.full, len(pca.PCA_VARS)
    rows = [
        ("Information reduction", f"{r.k80} of {p} components retain {res.cumulative[r.k80 - 1] * 100:.1f}% of the variance.",
         "Compact input for clustering and machine learning."),
        ("Dominant dimension", f"PC1 ({res.ratio[0] * 100:.1f}%): {r.interp['PC1'][1].lower()}.",
         "Primary axis for regime segmentation."),
        ("Seasonality removed", f"Raw PC1 explains {r.raw.ratio[0] * 100:.1f}% vs {res.ratio[0] * 100:.1f}% on anomalies; "
         f"month explains {r.structure.loc['Month', 'PC1']:.1f}% of anomaly PC1.", "Segments reflect weather, not the calendar."),
        ("Unusual days", f"{int(r.flag.sum()):,} days beyond the chi-square limit.", "Extreme-event screening and anomaly labelling."),
        ("Persistence", f"Lag-1 autocorrelation up to {r.autocorr.max():.2f}.", "Temporal regimes and sequence-aware models."),
    ]
    if r.no_aod is not None:
        aod = res.loadings.loc[pca.AOD, r.pcs].abs().max()
        rows.insert(2, ("AOD contribution",
                        f"Max |loading| {aod:.2f}; loading similarity without AOD {r.similarity.min():.2f}-{r.similarity.max():.2f}.",
                        "Decide whether AOD stays in segmentation features."))
    return ["## Step 7 - Analytical insights", "",
            f"**Information reduction.** {p} variables reduce to {r.k80} components ({res.cumulative[r.k80 - 1] * 100:.1f}% retained).", "",
            f"**Data structure.** {'Some' if r.structure.to_numpy().max() >= GROUP_ETA_MIN else 'No discrete'} grouping structure by "
            "municipality, month, year or AOD availability; organisation is mainly continuous with persistent anomalies.", "",
            "**Analytical value.** The variables collapse into a few interpretable dimensions, and the PCA shows how much of "
            "the relationships found earlier is shared structure rather than independent signal.", "",
            "**Potential applications.** Segmentation of days into weather regimes, feature selection, anomaly screening.", "",
            md_table(pd.DataFrame(rows, columns=["Insight", "Evidence", "Potential_Use"])), ""], rows


def step8(r, label, why_txt):
    res = r.full
    return ["## Step 8 - Pattern discovery assessment", "",
            f"**Pattern understanding.** {r.k80} dimensions describe the multivariate structure; PC1-PC{r.k80} are interpreted in Step 4.", "",
            f"**Dimensionality reduction success.** {'Meaningful' if r.k80 <= REDUCTION_MAX * len(pca.PCA_VARS) else 'Limited'} reduction: "
            f"{len(pca.PCA_VARS)} -> {r.k80} components keeping {res.cumulative[r.k80 - 1] * 100:.1f}%.", "",
            "**Analytical opportunities.** Regime segmentation, anomaly screening and a smaller feature set for the pilot models.", "",
            "**Future analytical directions.** Intelligent Segmentation Lab, clustering, feature selection, machine learning, "
            "predictive analytics (evaluated with the existing walk-forward folds).", "",
            f"### Project classification: **{label}**", "", why_txt, ""]


def executive_summary(r, label, why_txt):
    res = r.full
    aod_txt = (f"AOD {r.X[pca.AOD].isna().mean() * 100:.1f}% missing and set to climatology (shrinks its loading), "
               if pca.has_aod() else "AOD excluded, ")
    return ["## Executive Summary", "",
            "**Principal components:** " + "; ".join(f"{pc} ({r.interp[pc][1].lower()})" for pc in r.pcs) + ".", "",
            "**Explained variance:** " + ", ".join(f"{pc} {res.ratio[i] * 100:.1f}%" for i, pc in enumerate(r.pcs)) +
            f"; cumulative {res.cumulative[r.k80 - 1] * 100:.1f}%; {r.k80} components reach 80%.", "",
            "**Main patterns discovered:** " + "; ".join(f"{a.lower()}: {f}" for a, f in r.patterns[:2] + r.patterns[4:6]), "",
            f"**Dataset limitations:** only {r.distinct} distinct series among {r.df.municipio.nunique()} municipalities (NASA grid), "
            f"{aod_txt}zero-inflated rain, climatology estimated on the "
            "full sample, PCA captures linear structure only and is fitted on the full sample (k must be confirmed inside the walk-forward folds).", "",
            f"**Readiness for future activities: {label}.** {why_txt}", ""]


def build_report(r, n_days, files):
    label, why_txt = classify(r)
    s3, summary = step3(r)
    s4, interp_rows, loads, sens = step4(r)
    s5, dash = step5(r, files)
    s6, r.patterns = step6(r)
    s7, insights = step7(r)
    log_txt = (f" Variables with log1p before the anomaly: {', '.join(pca.LOG1P[r.profile])}." if pca.LOG1P[r.profile] else "")
    head = ["# Data Intelligence Lab - Pattern Discovery (Steps 1 to 8)", "", "## Dataset documentation", "",
            f"- **Dataset source:** {DATASET_SOURCE}", f"- **Dataset version:** {DATASET_VERSION}",
            f"- **Date of analysis:** {date.today().isoformat()}", f"- **Objective:** {OBJECTIVE}", f"- **Team members:** {TEAM_MEMBERS}",
            f"- **Rows / municipalities / days:** {len(r.df):,} / {r.df.municipio.nunique()} / {n_days:,}",
            f"- **PCA profile:** {r.profile} ({len(pca.PCA_VARS)} variables: {', '.join(pca.PCA_VARS)})", "",
            "Method note: PCA runs on deseasonalised anomalies (day-of-year climatology per municipality), standardised, on the "
            f"full master table.{log_txt}", ""]
    lines = head + step1(r) + step2(r) + s3 + s4 + s5 + s6 + s7 + step8(r, label, why_txt) + executive_summary(r, label, why_txt)
    tables = {"step3_pca_summary": summary, "step4_loadings": loads, "step4_interpretation": pd.DataFrame(interp_rows),
              "step5_dashboard": pd.DataFrame(dash), "step6_structure": r.structure.round(2),
              "step7_insights": pd.DataFrame(insights, columns=["Insight", "Evidence", "Potential_Use"])}
    if sens is not None:
        tables["step4_aod_sensitivity"] = sens
    return "\n".join(lines), tables, label


def run(profile: str, df0: pd.DataFrame, n_days: int, out_root: Path) -> str:
    pca.set_profile(profile)
    out = out_root / profile
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = df0.copy()
    for c in pca.LOG1P[profile]:                       # log1p ANTES de calcular anomalías
        df[c] = np.log1p(df[c].clip(lower=0))
    r = analyse(df, build_anomalies(df, pca.PCA_VARS), profile)
    files = {"Explained variance": pca.fig_variance(r.full, fig_dir / "fig_step5_explained_variance.png"),
             "Component loadings": pca.fig_loadings(r.full, fig_dir / "fig_step5_loadings.png", n=r.k80),
             "Two-component projections": pca.fig_projections(r.full, r.flag, fig_dir / "fig_step5_projections.png", n=r.k80),
             "PC1 vs PC2 by month": pca.fig_by_month(r.full, r.meta.month, fig_dir / "fig_step5_pc1_pc2_month.png")}
    report, tables, label = build_report(r, n_days, files)
    for name, t in tables.items():
        t.to_csv(tab_dir / f"{name}.csv", index=False)
    scores = r.meta[["date", "municipio"]].join(r.full.scores.iloc[:, :r.k80])
    scores.to_parquet(tab_dir / "pca_scores.parquet", index=False)
    (out / "data_analysis_report.md").write_text(report, encoding="utf-8")
    print(f"[{profile}] {len(pca.PCA_VARS)} vars -> k80 = {r.k80} "
          f"({r.full.cumulative[r.k80 - 1] * 100:.1f}%) | {label} | {out}")
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", choices=[*pca.PROFILES, "ambos"], default=PROFILE)
    a = ap.parse_args()
    df0, n_days = load_data(Path(a.data) if a.data else default_master_path())
    profiles = list(pca.PROFILES) if a.profile == "ambos" else [a.profile]
    for p in profiles:
        run(p, df0, n_days, Path(a.out) if a.out else OUT_DIR)


if __name__ == "__main__":
    main()