import cfg, common as c


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return v


def main():
    A, M, dash = c.load_json("assessment.json"), c.load_json("metrics.json"), c.load_json("dashboard.json")
    mrows = [dict(target=n, metric=k, value=fmt(v, 3) if isinstance(v, float) else v)
             for n, m in M.items() for k, v in m.items() if isinstance(v, (int, float, str))]
    out = ["# Explainable AI - Aguascalientes environmental forecasting (exp_ai)", "",
           "## Step 1 - Model selected", cfg.DL_VERDICT, "",
           f"Models: XGBoost, global, h={cfg.H}. **Scope: this report explains a single fold** (test year {cfg.TEST_YEAR}, "
           f"train < {cfg.TEST_YEAR} purged by H={cfg.H}); dl_validation's conclusions rest on four folds, so these "
           f"explanations are not a substitute for that evidence.", "",
           c.md_table(["target", "metric", "value"], mrows), "", "## Step 2 - SHAP analysis"]
    shap_j = {n: c.load_json(f"shap_{n}.json") for n in cfg.TARGETS}
    lime_j = {n: c.load_json(f"lime_{n}.json") for n in cfg.TARGETS}
    for n, s in shap_j.items():
        loc = [{**r, "pred": fmt(r["pred"]), "actual": fmt(r["actual"])} for r in s["local"]]
        out += [f"### {n} - SHAP Interpretation Matrix (units: {s['unit']})",
                c.md_table(["feature", "shap_contribution", "interpretation"], s["matrix"]), "",
                "Positive vs negative contributions: " + "; ".join(
                    f"{r['feature']} positive in {r['positive_share']:.0%} of days" for r in s["matrix"][:5]), "",
                f"![summary](images/{n}_shap_summary.png)", f"![importance](images/{n}_shap_importance.png)", "",
                "**Local explanations**",
                c.md_table(["id", "group", "date", "municipio", "pred", "actual", "top_positive", "top_negative"], loc), ""]

    out += ["## Step 3 - LIME analysis"]
    for n, l in lime_j.items():
        out += [f"### {n} - LIME Explanation Matrix", c.md_table(["observation", "key_factors", "interpretation"], l["matrix"]), ""]

    out += ["## Step 4 - Explainability dashboard"]
    for d in dash:
        out += [f"### [{d['target']}] {d['title']}", f"![{d['title']}]({d['path']})", "",
                f"- **What is explained:** {d['what']}", f"- **Key observations:** {d['obs']}",
                f"- **Practical relevance:** {d['practical']}", ""]

    sections = [("Step 5 - Transparency Assessment Matrix", "step5", ["target", "criterion", "assessment", "evidence"]),
                ("Step 6 - Explainability Insights Matrix", "step6", ["target", "insight", "evidence", "practical_meaning"]),
                ("Step 7 - Explainable AI Assessment Matrix", "step7", ["target", "criterion", "assessment", "evidence"])]
    for title, k, h in sections:
        out += [f"## {title}", c.md_table(h, A[k]), ""]

    out += ["## Step 8 - Explainability Recommendation", f"**{A['recommendation']}** (mean criterion score {A['mean_score']:.2f} / 2)", "",
            "## Executive Summary"]
    for n in cfg.TARGETS:
        s, l = shap_j[n], lime_j[n]
        drivers = ", ".join(r["feature"] for r in s["matrix"][:3])
        out += [f"### {n}",
                f"- **Model explained:** XGBoost ({cfg.TARGETS[n]['kind']}) with walk-forward-tuned parameters, h={cfg.H}, single fold (test year {cfg.TEST_YEAR}).",
                f"- **SHAP findings:** top drivers {drivers}; top-5 features hold {s['top5_mass']:.0%} of impact; ranking stability rho={s['stability_rho']:.2f}.",
                f"- **LIME findings:** {l['coherent_cases']}/{l['n_cases']} cases coherent; top-5 overlap with SHAP {l['mean_overlap']:.0%}, sign agreement {l['mean_sign_agreement']:.0%}.",
                f"- **Key predictive drivers:** {drivers}.",
                "- **Transparency assessment:** " + ", ".join(f"{k}={v}" for k, v in A["levels"][n].items()) + "."]
    out += ["", f"**Explainability recommendation: {A['recommendation']}**"]
    (cfg.OUT / "exp_ai.md").write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()