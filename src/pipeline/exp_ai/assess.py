import cfg, common as c

R = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
level = lambda v, hi, mid: "HIGH" if v >= hi else "MEDIUM" if v >= mid else "LOW"
NAMES = dict(transparency="Transparency", consistency="Consistency", interpretability="Interpretability",
             trustworthiness="Trustworthiness", stakeholder="Stakeholder Readiness", ethics="Ethical Considerations")
STATIC_PREFIX = ("geom_", "elev", "dist_")
PERSIST_CLF_SHARE = 0.10  # classifier: same-day precipitation share of total impact that flags persistence
SAME_DAY_PRECIP = ("precipitation_mm", "PRECTOTCORR", "dom_precip_mean", "dom_prectot_mean", "dom_precip_wet", "dom_prectot_wet")

def sign_flips(matrix):
    """Features whose own direction is opposite to their lag's direction (correlated-feature attribution)."""
    rho = {r["feature"]: r["rho"] for r in matrix if r["rho"] is not None and abs(r["rho"]) >= 0.15}
    out = []
    for f, v in rho.items():
        if "_lag" in f or "_mean" in f:
            continue
        for g, w in rho.items():
            if g.startswith(f + "_lag") and v * w < 0:
                out.append(f"{f}({v:+.2f}) vs {g}({w:+.2f})")
    return out


def per_target(name):
    s, l, m = c.load_json(f"shap_{name}.json"), c.load_json(f"lime_{name}.json"), c.load_json("metrics.json")[name]
    kind = cfg.TARGETS[name]["kind"]
    src = cfg.TARGETS[name]["src"]
    exact = s["additivity_err"] < 1e-3
    agree = min(l["mean_overlap"], l["mean_sign_agreement"])

    transp = "HIGH" if exact and s["stability_rho"] >= 0.8 else "MEDIUM" if exact else "LOW"
    cons = "HIGH" if agree >= 0.6 and s["stability_rho"] >= 0.8 else "MEDIUM" if agree >= 0.4 and s["stability_rho"] >= 0.6 else "LOW"
    interp = level(s["top5_mass"], 0.5, 0.3)
    skill_ok = m["bss"] > 0 if kind == "clf" else m.get("skill_vs_persistence", m["r2"]) > 0
    calib_ok = m["ece"] <= 0.05 if kind == "clf" else True

    # ---- flags first: ethical (bias / proxy features) vs model-quality (Step 6 risks)
    eth_flags, flags = [], []
    if s["bias_ratio"] >= 1.5:
        eth_flags.append(f"error differs {s['bias_ratio']:.1f}x across municipalities")
    static = [f for f in s["importance"] if f.startswith(STATIC_PREFIX)]
    tot = max(sum(s["importance"].values()), 1e-12)
    static_share = sum(s["importance"][f] for f in static) / tot
    if static and static_share > 0.03:
        eth_flags.append(f"static municipality identifiers act as features ({static_share:.0%} of impact: {', '.join(static[:3])})")
    if s["top1_share"] > 0.4:
        flags.append(f"single-driver dependence ({s['top1_share']:.0%} on one feature)")
    if not calib_ok:
        flags.append(f"probabilities miscalibrated (ECE {m['ece']:.3f})")
    if cons == "LOW":
        flags.append("SHAP and LIME disagree")
    if kind == "clf":
        pcols = [f for f in SAME_DAY_PRECIP if f in s["importance"]]
        src_share = sum(s["importance"][f] for f in pcols) / tot
        persist = src_share >= PERSIST_CLF_SHARE
        label = " + ".join(pcols) or src
    else:
        src_share = s["importance"].get(src, 0) / tot
        persist = s["matrix"][0]["feature"] == src
        label = src
    if persist:
        flags.append(f"same-day {label} carries {src_share:.0%} of impact: model leans on persistence")
    flips = sign_flips(s["matrix"])
    if flips:
        flags.append("opposite directions for a feature and its lag (correlated-feature attribution, not independent effects): "
                     + "; ".join(flips[:2]))
    all_flags = eth_flags + flags

    # ---- levels that depend on flags
    stake = level(s["top5_mass"], 0.6, 0.4) if s["top10_mass"] >= 0.5 else "LOW"
    capped = persist and stake == "HIGH"
    if capped:
        stake = "MEDIUM"
    trust = ("HIGH" if skill_ok and calib_ok and s["bias_ratio"] < 1.5 and cons != "LOW" and not all_flags
             else "MEDIUM" if skill_ok else "LOW")
    # licences are unverified (README), so ethics can never be HIGH
    ethics = "MEDIUM" if len(eth_flags) <= 1 else "LOW"

    perf = (f"BSS {m['bss']:.2f} (vs smoothed climatology), AUC {m['auc']:.2f}, ECE {m['ece']:.3f}" if kind == "clf"
            else f"R2 {m['r2']:.2f}, skill vs persistence {m.get('skill_vs_persistence', float('nan')):.2f}")
    rc = l["rank_corr_with_shap"]
    ev = dict(
        transparency=f"SHAP additivity error {s['additivity_err']:.1e}; top-feature ranking stability (2 half-years) rho={s['stability_rho']:.2f}",
        consistency=f"SHAP-LIME top-5 overlap {l['mean_overlap']:.0%}, sign agreement {l['mean_sign_agreement']:.0%}, "
                    f"{l['coherent_cases']}/{l['n_cases']} coherent cases, rank corr {'n/a' if rc is None else f'{rc:.2f}'}",
        interpretability=f"top-5 features hold {s['top5_mass']:.0%} of impact, top-10 {s['top10_mass']:.0%}; {m['n_features']} features in total",
        trustworthiness=f"{perf}; max/min municipal error ratio {s['bias_ratio']:.2f}",
        stakeholder=f"top-5 mass {s['top5_mass']:.0%}, top-10 mass {s['top10_mass']:.0%} of {m['n_features']} features"
                    + ("; capped at MEDIUM: persistence flag raised" if capped else ""),
        ethics=("; ".join(eth_flags) if eth_flags else "no bias/proxy flags")
               + "; municipal-scale data (no personal data); scope limited to h=1, 11 municipalities; "
                 "source licences unverified (caps rating at MEDIUM)")
    return dict(levels=dict(transparency=transp, consistency=cons, interpretability=interp, trustworthiness=trust,
                            stakeholder=stake, ethics=ethics), evidence=ev, flags=all_flags, s=s, l=l, m=m)


def main():
    res = {n: per_target(n) for n in cfg.TARGETS}
    step5, step6, step7 = [], [], []
    for n, r in res.items():
        L, E = r["levels"], r["evidence"]
        for k in ("transparency", "consistency", "interpretability", "trustworthiness"):
            step5.append(dict(target=n, criterion=NAMES[k], assessment=L[k], evidence=E[k]))
        for k in ("interpretability", "transparency", "stakeholder", "ethics"):
            step7.append(dict(target=n, criterion=NAMES[k], assessment=L[k], evidence=E[k]))
        top = r["s"]["matrix"][:3]
        agg = r["l"]["aggregate_importance"]
        lime_top = sorted(agg, key=agg.get, reverse=True)[:3]
        step6 += [
            dict(target=n, insight="Key predictive drivers",
                 evidence="SHAP: " + ", ".join(x["feature"] for x in top) + "; LIME (aggregate): " + ", ".join(lime_top),
                 practical_meaning="These inputs must be monitored first; data failures here degrade forecasts most."),
            dict(target=n, insight="Decision logic",
                 evidence="; ".join(f"{x['feature']}: {x['interpretation'].split(' (')[0]}" for x in top),
                 practical_meaning="Check these directions against meteorological expectations before deployment; "
                                   "directions are attributions among correlated features, not isolated causal effects."),
            dict(target=n, insight="Risks / weaknesses", evidence="; ".join(r["flags"]) or "no automatic flags",
                 practical_meaning="Flags mark where explanations reveal bias, over-reliance or instability; "
                                   "address them before operational use."),
        ]
    all_lv = [v for r in res.values() for v in r["levels"].values()]
    mean = sum(R[v] for v in all_lv) / len(all_lv)
    anyLow = "LOW" in all_lv
    rec = ("EXPLAINABLE AND READY" if mean >= 1.5 and not anyLow
           else "PARTIALLY EXPLAINABLE" if mean >= 0.75 else "ADDITIONAL EXPLAINABILITY REQUIRED")
    c.save_json(dict(step5=step5, step6=step6, step7=step7, recommendation=rec, mean_score=mean,
                     levels={n: r["levels"] for n, r in res.items()}), "assessment.json")


if __name__ == "__main__":
    main()