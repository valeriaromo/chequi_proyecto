import json
import cfg
from cfg import P

def rd(name):
    p = cfg.OUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f(x, n=3):
    return "n/a" if x is None or x != x else f"{x:.{n}f}"


def main():
    leak, S, gate, xai = rd("leakage.json"), rd("benchmark_summary.json"), rd("gate.json"), rd("explain.json")
    L = ["# Deep Learning Validation – ChequiProyecto", ""]

    L += ["## 1. Leakage and integrity checks", ""]
    if leak:
        L += [md(["Check", "Result"], [
            ["Date gaps per municipality (breaks shift(-h))", leak["grid_gaps"]],
            ["Target alignment mismatches", ", ".join(f"{t}: {v['mismatches']}/{v['compared']}" for t, v in leak["alignment"].items())],
            ["Raw NaN in target source columns (targets built after ffill)", ", ".join(f"{t}: {n}" for t, n in leak["raw_nan_in_target_cols"].items())],
            ["Placebo (shuffled targets, test " + str(cfg.PLACEBO_YEAR) + ")",
             ", ".join(f"{t}: {[(k, round(v, 3)) for k, v in m.items() if k in ('bss', 'r2')][0]}" for t, m in leak["placebo"].items() if t != "passed")
             + (" – passed" if leak["placebo"]["passed"] else " – FAILED")],
            ["Overall", "PASSED" if leak["passed"] else "FAILED"]]), ""]
        L += [f"- {w}" for w in leak["warnings"]] + [""]

    L += ["## 2. Fair benchmark (multi-seed, same baseline)", "",
          f"{len(cfg.SEEDS)} seeds × {len(P.TEST_YEARS)} walk-forward folds. Classification: BSS vs smoothed climatology. "
          f"Regression: skill = 1 − {cfg.SKILL_METRIC.upper()} / best(persistence, smoothed climatology).", ""]
    if S:
        for t, by_cfg in S.items():
            years = list(next(iter(by_cfg.values()))["per_year"])
            rows = [[c, f(e["mean"]), f(e["seed_std"])] + [f(e["per_year"][y]) for y in years] for c, e in by_cfg.items()]
            if gate and "xgb_per_year" in gate.get(t, {}):
                xs = gate[t]["xgb_per_year"]
                rows.append(["XGBoost (frozen)", f(sum(xs.values()) / len(xs)), "–"] + [f(xs[y]) for y in years])
            L += [f"**{t}**", "", md(["Config", "Mean", "Seed std"] + years, rows), ""]
            clf = [(c, e) for c, e in by_cfg.items() if "ece" in e]
            if clf:
                L += [md(["Config", "ROC AUC", "ECE", "Tuned thr", "Precision", "Recall", "F1"],
                         [[c, f(e["auc"]), f(e["ece"]), f(e["thr"], 2), f(e["precision"]), f(e["recall"]), f(e["f1"])] for c, e in clf]),
                      "", "Calibration curve: `figures/calibration.png`; thresholds tuned on the validation year.", ""]

    L += ["## 3. Gate and recommendation per target", ""]
    if gate:
        L += [md(["Target", "Best config", "DL mean", "Mean gain vs XGBoost", "Threshold", "Wins", "Multitask − single", "Verdict"],
                 [[t, g["best_config"], f(g["dl_mean"]), f(g.get("mean_gain")), f(g.get("threshold")), g.get("wins", "n/a"),
                   f(g.get("multitask_minus_single")), g["verdict"]] for t, g in gate.items()]), "",
              f"Rule: DL is recommended only if gain > max({cfg.MIN_GAIN}, {cfg.NOISE_K}×seed std) in every fold and DL skill is positive in every fold. "
              "Ties go to XGBoost.", ""]

    L += ["## 4. Explainability", ""]
    if xai and "skipped" in xai:
        L += [f"Skipped: {xai['skipped']}.", ""]
    elif xai:
        L += [f"Model trained on {', '.join(xai['targets'])}, test {xai['year']}, seed {xai['seed']}."
              + (" **Diagnostic only: no target passed the gate.**" if xai["forced"] else ""), ""]
        for t in xai["targets"]:
            x = xai[t]
            L += [f"**{t}**", "", md(["Feature", "Loss increase", "Std"], [[d["feature"], f(d["increase"], 4), f(d["std"], 4)] for d in x["top_features"]]), "",
                  md(["Lag block / group", "Loss increase"], [[k, f(v["increase"], 4)] for k, v in {**x["lag_blocks"], **x["groups"]}.items()]), "",
                  f"Spearman between permutation and gradient×input rankings: {f(x['spearman_vs_gradxinput'], 2)}. "
                  f"Figure: `figures/xai_{t}.png`. Loss by month and municipality are in `explain.json`.", ""]
        L += ["Caveat: lags and rolling means are collinear, so single-feature importance is understated; read the group and lag-block rows first.", ""]

    L += ["## 5. Next step", ""]
    if leak and not leak["passed"]:
        L += ["Fix the failed integrity checks before using any result above."]
    elif gate and any(g["eligible"] for g in gate.values()):
        ok = [t for t, g in gate.items() if g["eligible"]]
        L += [f"Extend DL only to targets that pass the gate (currently: {', '.join(ok)}). Add each new variable to `TARGETS` and `cfg.BASELINE`, "
              "rerun `run_validation.py`, and keep it only if it passes."]
    else:
        L += ["No target passed the gate: keep the XGBoost pilot and do not extend DL to more variables."]
    (cfg.OUT_DIR / "dl_validation.md").write_text("\n".join(L), encoding="utf-8")