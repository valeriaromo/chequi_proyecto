import json
import pandas as pd
import model_config as cfg
import plots
from dataset import feature_group

LEVELS = {"Low": 1, "Medium": 2, "High": 3}
PRIMARY = {"classification": ("roc_auc", True), "regression": ("mae", False)}  # rankear por AUC alto y MAE bajo
TIE_TOLERANCE = 0.02   # si dos modelos están dentro de 2% del mejor, desempatar por complejidad
AUC_USEFUL, AUC_STRONG = 0.70, 0.80
SKILL_USEFUL, SKILL_STRONG = 0.0, 0.10  # el skill es la mejora de MAE sobre la persistencia; 0.10 = 10% mejor
CLF_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REG_METRICS = ["mae", "mse", "rmse", "r2"]
LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC AUC", "mae": "MAE", "mse": "MSE", "rmse": "RMSE", "r2": "R²"}
TASK_TITLES = {
    "classification": f"Classification – next-day rain occurrence (≥ {cfg.RAIN_THRESHOLD_MM:g} mm, h={cfg.HORIZON})",
    "regression": f"Regression – next-day minimum temperature ({cfg.TEMP_COL}, h={cfg.HORIZON})",
}
FEATURE_DESC = {
    "base": "all numeric columns of the table (AOD excluded)",
    "aod": "base + AOD columns",
    "static": "base + static municipal columns (area, population, housing); AOD excluded",
    "all": "base + static + cyclone declaration columns; AOD excluded",
}
GROUP_DESC = {
    "base": "Table columns (weather, lags, rolling means, other)",
    "static": "Static municipal columns",
    "cyclone": "Cyclone declaration columns",
    "aod": "AOD columns",
}
HEATMAP_TOP = 20
TABLE_TOP = 15


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def load_results(out) -> dict:
    tasks = {"classification": [], "regression": []}
    for path in sorted(out.glob("*_summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["folds"] = pd.read_csv(out / f"{summary['slug']}_folds.csv")
        summary["importance"] = pd.read_csv(out / f"{summary['slug']}_importance.csv")
        tasks[summary["task"]].append(summary)
    return {task: group for task, group in tasks.items() if group}


def primary(s) -> float:
    return s["pooled"][PRIMARY[s["task"]][0]]


def skill(s) -> float:
    # el skill es la mejora de MAE sobre la persistencia; 0.10 = 10% mejor
    return 1 - s["pooled"]["mae"] / s["baseline"]["persistence_mae"]


def top_performer(group):
    higher = PRIMARY[group[0]["task"]][1]
    return (max if higher else min)(group, key=primary)


def gap_to_top(s, group) -> float:
    top, higher = primary(top_performer(group)), PRIMARY[s["task"]][1]
    return ((top - primary(s)) if higher else (primary(s) - top)) / abs(top)


def select_best(group):
    # si hay empate en la métrica primaria, desempatar por complejidad (Low < Medium < High)
    candidates = [s for s in group if gap_to_top(s, group) <= TIE_TOLERANCE]
    return min(candidates, key=lambda s: (LEVELS[s["complexity"]], gap_to_top(s, group)))


def fold_cv(s) -> float:
    values = s["folds"][PRIMARY[s["task"]][0]]
    return values.std() / abs(values.mean())


def performance(s) -> str:
    p = s["pooled"]
    return f"AUC {p['roc_auc']:.2f}, F1 {p['f1']:.2f}" if s["task"] == "classification" else f"MAE {p['mae']:.2f}, R² {p['r2']:.2f}"


def assessment(s) -> str:
    p = s["pooled"]
    if s["task"] == "classification":
        auc = p["roc_auc"]
        label = "Good" if auc >= AUC_STRONG else "Moderate" if auc >= AUC_USEFUL else "Weak" if auc >= 0.60 else "No useful"
        return f"{label} ranking skill; recall {p['recall']:.2f}, precision {p['precision']:.2f}"
    sk = skill(s)
    label = "Clearly beats" if sk >= SKILL_STRONG else "Marginally beats" if sk > SKILL_USEFUL else "Does not beat"
    return f"{label} persistence ({sk:+.0%} MAE)"


def task_checks(task, best):
    folds = best["folds"]
    if task == "classification":
        auc = best["pooled"]["roc_auc"]
        return auc >= AUC_USEFUL, auc >= AUC_STRONG, bool((folds["roc_auc"] >= AUC_USEFUL).all())
    sk = skill(best)
    return sk > SKILL_USEFUL, sk >= SKILL_STRONG, bool((folds["mae"] < folds["persistence_mae"]).all())


def readiness(tasks, best) -> str:
    checks = [task_checks(t, best[t]) for t in tasks]
    if all(all(c) for c in checks):
        return "READY FOR MODEL VALIDATION"
    if any(c[0] for c in checks):
        return "CONDITIONALLY READY"
    return "ADDITIONAL DATA REQUIRED"


def params_text(s) -> str:
    return ", ".join(f"{k}={v}" for k, v in s["params"].items())


def _first(tasks):
    return next(iter(tasks.values()))[0]


def _features_of(tasks) -> list:
    first = _first(tasks)
    return first.get("features") or list(first["importance"]["feature"])


def importance_matrix(group) -> pd.DataFrame:
    """Filas = features, columnas = modelos, valores = permutation importance; ordenado por media."""
    frames = [s["importance"].set_index("feature")["importance_mean"].rename(s["name"]) for s in group]
    m = pd.concat(frames, axis=1)
    m["mean"] = m.mean(axis=1)
    return m.sort_values("mean", ascending=False)


def group_importance(matrix) -> pd.DataFrame:
    m = matrix.drop(columns="mean").clip(lower=0)
    return m.groupby(m.index.map(feature_group)).sum()


def section_header(tasks) -> str:
    first = _first(tasks)
    fs = first.get("feature_set", "base")
    aod_text = "AOD columns are included" if fs == "aod" else "every AOD column is excluded"
    return "\n".join([
        "# Machine Learning Challenge",
        "",
        f"Source table: `{first['table']}` | feature set: `{fs}` ({FEATURE_DESC.get(fs, fs)}) | "
        f"rows per task: {first['n_rows']:,} | features: {first['n_features']}",
        "",
        "Scope (from the project README): 11 municipalities of Aguascalientes, daily data, horizon h=1. "
        "Targets are built with `groupby('municipio').shift(-h)`; features exclude `date`, `municipio`, `year`, "
        f"targets; {aod_text}. All outputs are generated by `run_all.py`.",
    ])


def section_features(tasks) -> str:
    feats = _features_of(tasks)
    groups = {}
    for f in feats:
        groups.setdefault(feature_group(f), []).append(f)
    rows = [(g, GROUP_DESC[g], len(cols), ", ".join(f"`{c}`" for c in cols[:6]) + (", …" if len(cols) > 6 else ""))
            for g, cols in groups.items()]
    return "\n".join([
        "## Feature Inventory", "",
        f"Every model is trained on the same {len(feats)} features (median-imputed and standardized per fold).", "",
        md_table(["Group", "Description", "Count", "Examples"], rows), "",
        "The complete list is stored in each `*_summary.json` under `features`.",
    ])


def section_training(tasks) -> str:
    models = [s for group in tasks.values() for s in group]
    out = [
        "## Step 3 – Machine Learning Model Training", "",
        "### Model Training Matrix", "",
        md_table(["Model", "Type", "Main Parameters"],
                 [(s["name"], f"{s['model_type']} ({s['task']})", params_text(s)) for s in models]),
        "", "### Training process", "",
        "- Validation: expanding-window walk-forward by year; each test year is predicted by a model trained on all earlier years "
        f"(folds {', '.join(str(y) for y in cfg.FOLD_YEARS)}; 2026 is a partial year).",
        "- Each fold fits a fresh pipeline: median imputation → standardization → model, using training rows only.",
        "- No hyperparameter tuning: parameters are fixed in each model script so the test folds are never used for selection.",
        "- Regression targets are standardized inside SVR and the MLP regressor. SVM uses balanced class weights.",
        "", md_table(["Model", "Features", "Train rows (min–max)", "Total fit time (s)"],
                    [(s["name"], s["n_features"], f"{s['folds']['n_train'].min():,}–{s['folds']['n_train'].max():,}",
                      f"{s['fit_seconds']:.1f}") for s in models]),
        "", "### Initial observations", "",
    ]
    for task, group in tasks.items():
        fastest, slowest = min(group, key=lambda s: s["fit_seconds"]), max(group, key=lambda s: s["fit_seconds"])
        out.append(f"- {task.title()}: fastest to train {fastest['name']} ({fastest['fit_seconds']:.1f} s), "
                   f"slowest {slowest['name']} ({slowest['fit_seconds']:.1f} s).")
        if task == "classification":
            out.append(f"- Rain occurs on {group[0]['baseline']['positive_rate']:.1%} of test days, so accuracy alone is misleading; "
                       "ROC AUC and recall carry the reading.")
            low = [s["name"] for s in group if s["pooled"]["recall"] < 0.5]
            if low:
                out.append(f"- Recall is below 0.50 at the default threshold for: {', '.join(low)}.")
        else:
            failing = [s["name"] for s in group if skill(s) <= SKILL_USEFUL]
            out.append(f"- Models that do not beat the persistence baseline: {', '.join(failing)}." if failing
                       else "- All regression models beat the persistence baseline.")
    return "\n".join(out)


def section_evaluation(tasks) -> str:
    out = ["## Step 4 – Model Evaluation", "", "Metrics are pooled over all walk-forward test folds.", ""]
    for task, group in tasks.items():
        columns = CLF_METRICS if task == "classification" else REG_METRICS
        out += [f"### {TASK_TITLES[task]}", "", "#### Model Evaluation Matrix", "",
                md_table(["Model"] + [LABELS[c] for c in columns] + ["Overall Assessment"],
                         [(s["name"], *(f"{s['pooled'][c]:.3f}" for c in columns), assessment(s)) for s in group]), ""]
        if task == "classification":
            out += ["![ROC overlay](clf_roc_overlay.png)", ""]
        for s in group:
            images = ["confusion_matrix", "roc"] if task == "classification" else ["pred_vs_actual"]
            out.append(f"**{s['name']}** " + " ".join(f"![{s['name']} {i}]({s['slug']}_{i}.png)" for i in images))
            out.append("")
    return "\n".join(out)


def section_feature_evaluation(tasks, matrices) -> str:
    out = ["## Feature Evaluation", "",
           "Permutation importance of every feature, computed on the last walk-forward fold "
           f"({_first(tasks)['n_features']} features). Units: ROC AUC drop (classification) or MAE increase (regression). "
           "Negative values mean shuffling the feature did not hurt (noise).", ""]
    for task, group in tasks.items():
        m = matrices[task]
        names = [s["name"] for s in group]
        out += [f"### {TASK_TITLES[task]}", "",
                f"![{task} importance heatmap]({task}_importance_heatmap.png)", "",
                f"**Top {TABLE_TOP} features across models**", "",
                md_table(["Feature", "Group"] + names + ["Mean"],
                         [(f"`{f}`", feature_group(f), *(f"{r[n]:.4f}" for n in names), f"{r['mean']:.4f}")
                          for f, r in m.head(TABLE_TOP).iterrows()]), "",
                "**Importance by feature group (sum of positive importances)**", "",
                md_table(["Group"] + names,
                         [(g, *(f"{r[n]:.4f}" for n in names)) for g, r in group_importance(m).iterrows()]), "",
                f"Full matrix for all features: `{task}_feature_importance_matrix.csv`.", ""]
        useless = [f for f, v in m["mean"].items() if v <= 0]
        out += [f"{len(useless)} of {len(m)} features have mean importance ≤ 0 across models.", ""]
    return "\n".join(out)


def section_comparison(tasks, best) -> str:
    out = ["## Phase 3 – Model Comparison", "", "### Step 5 – Comparative Analysis", ""]
    for task, group in tasks.items():
        metric = PRIMARY[task][0]
        out += [f"#### {TASK_TITLES[task]}", "", f"**Stability** (per-fold {LABELS[metric]}; lower CV = more consistent)", "",
                md_table(["Model", "Mean", "Std", "CV", "Worst fold"],
                         [(s["name"], f"{s['folds'][metric].mean():.3f}", f"{s['folds'][metric].std():.3f}",
                           f"{fold_cv(s):.1%}", _worst_fold(s)) for s in group]), ""]
        out += ["**Model Complexity Assessment**", "",
                md_table(["Model", "Train time (s)", "Predict time (s)", "Data requirements", "Deployment difficulty"],
                         [(s["name"], f"{s['fit_seconds']:.1f}", f"{s['predict_seconds']:.2f}", s["data_needs"], s["deployment"]) for s in group]), ""]
        out += ["**Model Comparison Matrix**", "",
                md_table(["Model", "Performance", "Complexity", "Interpretability", "Recommendation"],
                         [(s["name"], performance(s), s["complexity"], s["interpretability"], _recommendation(s, best[task], group)) for s in group]), "",
                "**Model Practicality Matrix**", "",
                md_table(["Model", "Performance", "Complexity", "Interpretability", "Deployment Difficulty"],
                         [(s["name"], performance(s), s["complexity"], s["interpretability"], s["deployment"]) for s in group]), ""]
    return "\n".join(out)


def _worst_fold(s) -> str:
    metric, higher = PRIMARY[s["task"]]
    row = s["folds"].loc[s["folds"][metric].idxmin() if higher else s["folds"][metric].idxmax()]
    return f"{row[metric]:.3f} ({int(row['fold'])})"


def _recommendation(s, best, group) -> str:
    if s is best:
        return "**Selected**"
    gap = gap_to_top(s, group)
    if gap <= TIE_TOLERANCE:
        return "Comparable performance, higher complexity" if LEVELS[s["complexity"]] > LEVELS[best["complexity"]] else "Comparable alternative"
    return f"Not recommended ({gap:.0%} worse on {PRIMARY[s['task']][0].upper()})"


def section_selection(tasks, best) -> str:
    out = ["### Step 6 – Best Model Selection", "", "#### Best Model Selection Report", "",
           f"Decision rule: best primary metric (ROC AUC for classification, MAE for regression); models within {TIE_TOLERANCE:.0%} "
           "of the best are ranked by complexity and the simplest wins."]
    for task, group in tasks.items():
        b = best[task]
        metrics = ", ".join(f"{LABELS[k]} {b['pooled'][k]:.3f}" for k in (CLF_METRICS if task == "classification" else REG_METRICS))
        data_line = (f"{b['n_rows']:,} rows, {b['n_features']} features, rain on {b['baseline']['positive_rate']:.1%} of test days"
                     if task == "classification" else
                     f"{b['n_rows']:,} rows, {b['n_features']} features, persistence MAE {b['baseline']['persistence_mae']:.3f}")
        out += ["", f"**{TASK_TITLES[task]} → {b['name']}**", "",
                f"- Performance metrics: {metrics}. {assessment(b)}.",
                f"- Project constraints: pilot scope is municipal, daily, h={cfg.HORIZON}; the project already monitors frozen models, "
                f"so a low-maintenance model with a single sklearn pipeline artifact is preferred (complexity {b['complexity']}, deployment {b['deployment']}).",
                f"- Data characteristics: {data_line}; lags and rolling means are strongly collinear, which penalizes Naive Bayes and OLS coefficient reading.",
                f"- Practical applicability: fit {b['fit_seconds']:.1f} s over {len(b['folds'])} folds, predict {b['predict_seconds']:.2f} s; "
                f"interpretability {b['interpretability']}."]
    return "\n".join(out)


def section_insights(tasks, best) -> str:
    out = ["## Phase 4 – Machine Learning Intelligence", "", "### Step 7 – Predictive Insights", "",
           "#### Model Behavior", ""]
    rows = []
    for task, group in tasks.items():
        b, top = best[task], best[task]["importance"].head(3)
        feats = ", ".join(f"`{r.feature}` ({r.importance_mean:.3f})" for r in top.itertuples())
        unit = "ROC AUC drop" if task == "classification" else "MAE increase"
        out.append(f"- {b['name']} ({task}): {assessment(b)}. Permutation importance on the last fold is led by {feats}.")
        rows.append((f"{b['name']} relies on a few inputs", f"Top-3 permutation importance ({unit}): {feats}",
                     "Quality checks and monitoring should start with these columns; a failure in them degrades predictions most."))
        useful = task_checks(task, b)[0]
        rows.append((f"{task.title()} skill is {'present' if useful else 'absent'}", assessment(b),
                     "Predictions carry information beyond the baseline." if useful else "Predictions should not be used operationally."))
        rows.append((f"{task.title()} stability across years", f"CV of {LABELS[PRIMARY[task][0]]}: {fold_cv(b):.1%}; worst fold {_worst_fold(b)}",
                     "A high CV means year-to-year performance is not guaranteed."))
    if "regression" in tasks:
        rows.append(_linearity_insight(tasks["regression"]))
    if "classification" in tasks:
        b = best["classification"]
        rows.append(("Default 0.5 threshold trades recall for precision",
                     f"{b['name']}: recall {b['pooled']['recall']:.2f}, precision {b['pooled']['precision']:.2f}",
                     "Operational use needs a threshold chosen for the cost of missed vs. false rain alerts; probability calibration is unverified."))
    out += ["", "#### Key Predictive Variables", ""]
    for task in tasks:
        b = best[task]
        out += [f"**{b['name']}**", "", f"![{b['name']} importance]({b['slug']}_importance.png)", "",
                md_table(["Feature", "Group", "Importance (mean ± std)"],
                         [(f"`{r.feature}`", feature_group(r.feature), f"{r.importance_mean:.4f} ± {r.importance_std:.4f}")
                          for r in b["importance"].head(10).itertuples()]), ""]
    out += ["#### Limitations", "",
            "- Only 4 walk-forward folds: confidence intervals are not credible and fold-level rankings can flip.",
            "- No hyperparameter tuning and no calibration check of rain probabilities.",
            "- 11 municipalities share weather systems, so rows of the same day are not independent.",
            "- Target construction has not been audited for leakage (the project audit covers features only).",
            "- Permutation importance understates features that have correlated substitutes (lags, rolling means); with the full feature set it can split credit among them.",
            "- Permutation importance is computed on one fold and a 1,000-row sample (3 repeats), so small values are noise.",
            "- Static and cyclone columns are constant per municipality (or per declaration), so they can only separate municipalities, not days.",
            "- MLP early stopping uses a random validation split inside the training years, not a temporal one.",
            "- Results describe model behavior only and are not official forecasts.",
            "", "#### Predictive Insights Matrix", "", md_table(["Insight", "Evidence", "Practical Meaning"], rows)]
    return "\n".join(out)


def _linearity_insight(group):
    linear = next((s for s in group if s["name"].startswith("Linear")), None)
    others = [s for s in group if s is not linear]
    if linear is None or not others:
        return ("Linearity of the signal", "No linear baseline in the results", "Run the linear model to compare.")
    best_other = min(others, key=primary)
    gain = (primary(linear) - primary(best_other)) / primary(linear)
    meaning = ("Most of the signal is linear (persistence, seasonality); complex models add little."
               if gain < TIE_TOLERANCE else "Nonlinear interactions carry extra signal; added complexity is justified.")
    return ("Linearity of the signal", f"Linear MAE {primary(linear):.3f} vs {best_other['name']} {primary(best_other):.3f} ({gain:+.1%})", meaning)


def section_assessment(tasks, best) -> str:
    verdict = readiness(tasks, best)
    rows = []
    for task in tasks:
        useful, strong, reliable = task_checks(task, best[task])
        rows.append((TASK_TITLES[task], best[task]["name"], "Yes" if useful else "No",
                     "Yes" if reliable else "No (some folds fail)",
                     "Could support decisions as a screening aid" if useful and reliable and strong
                     else "Only with caution" if useful else "No"))
    return "\n".join([
        "### Step 8 – Machine Learning Assessment", "", "#### Machine Learning Assessment and Analytical Roadmap", "",
        md_table(["Task", "Selected model", "Predictive capability", "Model reliability", "Practical value"], rows), "",
        "Criteria: useful = ROC AUC ≥ 0.70 or MAE below persistence; strong = AUC ≥ 0.80 or ≥ 10% MAE gain; "
        "reliable = every walk-forward fold passes the useful criterion.", "",
        f"**Classification of the project: {verdict}**", "",
        "#### Future Analytical Directions", "",
        md_table(["Direction", "Why"], [
            ("Model Performance Validation", "Calibrate rain probabilities, audit target leakage, add confidence intervals (more folds)."),
            ("Explainable AI Workshop", "Permutation importance is only a first look; add partial dependence and SHAP on the selected models."),
            ("Predictive Intelligence Engine", "Freeze the selected models and connect them to the existing rolling-skill monitoring."),
            ("Deep Learning Innovation Lab", "Only if the MLP clearly beats simpler models in the matrices above."),
            ("Intelligent Segmentation Lab", "Check whether municipalities cluster into regimes with different skill."),
        ]),
    ])


def section_summary(tasks, best) -> str:
    first = _first(tasks)
    models = [s["name"] for g in tasks.values() for s in g]
    lines = ["## Executive Summary", "",
             f"- **Feature set:** `{first.get('feature_set', 'base')}`, {first['n_features']} features "
             f"({FEATURE_DESC.get(first.get('feature_set', 'base'), '')}).",
             f"- **Models trained:** {', '.join(models)}."]
    for task in tasks:
        b = best[task]
        key = (f"ROC AUC {b['pooled']['roc_auc']:.3f}, F1 {b['pooled']['f1']:.3f}, recall {b['pooled']['recall']:.3f}"
               if task == "classification" else
               f"MAE {b['pooled']['mae']:.3f}, RMSE {b['pooled']['rmse']:.3f}, R² {b['pooled']['r2']:.3f}, {skill(b):+.0%} MAE vs persistence")
        lines.append(f"- **Best model – {task}:** {b['name']} ({key}).")
    lines.append("- **Key insights:** " + "; ".join(
        f"{best[t]['name']} is led by `{best[t]['importance'].iloc[0]['feature']}`" for t in tasks) + ".")
    lines.append("- **Limitations:** 4 walk-forward folds, no tuning, no calibration check, no target-leakage audit, correlated features.")
    lines.append(f"- **Readiness assessment:** **{readiness(tasks, best)}**.")
    return "\n".join(lines)


def write_report(table: str = "master", feature_set: str = "base"):
    out = cfg.output_dir(table, feature_set)
    tasks = load_results(out)
    if not tasks:
        raise SystemExit(f"No model outputs found in {out}. Run the model scripts first.")
    best = {task: select_best(group) for task, group in tasks.items()}
    tag = f"[{feature_set}, {_first(tasks)['n_features']} features]"

    if "classification" in tasks:
        curves = []
        for s in tasks["classification"]:
            p = pd.read_parquet(out / f"{s['slug']}_predictions.parquet")
            curves.append((s["name"], p["y_true"], p["y_score"]))
        plots.plot_roc_overlay(curves, out / "clf_roc_overlay.png", tag)

    matrices = {}
    for task, group in tasks.items():
        m = importance_matrix(group)
        matrices[task] = m
        m.to_csv(out / f"{task}_feature_importance_matrix.csv")
        plots.plot_importance_heatmap(m.drop(columns="mean"), f"{task.title()} – feature importance by model {tag}",
                                      out / f"{task}_importance_heatmap.png", top=HEATMAP_TOP)

    sections = [section_header(tasks), section_features(tasks), section_training(tasks), section_evaluation(tasks),
                section_feature_evaluation(tasks, matrices), section_comparison(tasks, best),
                section_selection(tasks, best), section_insights(tasks, best),
                section_assessment(tasks, best), section_summary(tasks, best)]
    path = out / "machine_learning_challenge.md"
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate the report from existing outputs")
    parser.add_argument("--table", choices=sorted(cfg.TABLES), default="master")
    parser.add_argument("--features", choices=cfg.FEATURE_SETS, default="base")
    a = parser.parse_args()
    print(write_report(a.table, a.features))