import numpy as np
import params as P
from evaluate import load_baseline

PRIMARY = {"clf": ("bss", True), "reg": ("mae", False)}  # (metric, higher_is_better)


def md(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] +
                     ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def compare(res):
    rows = []  # (target, year, dl, baseline, gain)
    # primary metrics are bss for classification, mae for regression; higher_is_better = True for bss, False for mae
    for t in P.ACTIVE_TARGETS:
        metric, hib = PRIMARY[P.TARGETS[t]["kind"]]
        base = load_baseline(t)
        if base is None or metric not in base.columns:
            continue
        for y, fo in res["folds"].items():
            if int(y) in base.index:
                dl, b = fo["metrics"][t][metric], base.loc[int(y), metric]
                rows.append((t, int(y), dl, b, (dl - b) if hib else (b - dl)))
    return rows


def decide(rows):
    if not rows:
        return "ADDITIONAL EXPERIMENTATION REQUIRED", 0, 0, 0.0
    gains = np.array([r[4] for r in rows])
    wins, mean = int((gains > 0).sum()), float(gains.mean())
    if wins == len(gains) and mean >= P.MIN_GAIN:
        return "DEEP LEARNING RECOMMENDED", wins, len(gains), mean
    if wins == 0 or mean <= 0:
        return "MACHINE LEARNING SUFFICIENT", wins, len(gains), mean
    return "ADDITIONAL EXPERIMENTATION REQUIRED", wins, len(gains), mean


def build(res):
    folds, years = res["folds"], list(res["folds"])
    nF, secs = res["n_features"], sum(fo["seconds"] for fo in folds.values())
    rows = compare(res)
    verdict, wins, n, gain = decide(rows)
    heads = ", ".join(f"{t} ({P.TARGETS[t]['kind']})" for t in P.ACTIVE_TARGETS)
    L = ["# Deep Learning Assessment – ChequiProyecto", "",
         "## Step 2 – Architecture Design", "",
         md(["Component", "Configuration"], [
             ["Input Layer", f"{P.WINDOW}-day window × {nF} features per municipality, sequence-to-one, horizon h={P.HORIZON}"],
             ["Hidden Layers", f"LSTM ×{P.LSTM_LAYERS} ({P.HIDDEN} units) → Dropout({P.DROPOUT}) → Dense({P.DENSE})"],
             ["Output Layer", f"One Linear(1) head per target: {heads}"],
             ["Activation Functions", "LSTM tanh/sigmoid gates; Dense ReLU; classification head sigmoid (on logit); regression head linear"]]), "",
         "## Step 3 – Model Training", "",
         md(["Parameter", "Value"], [
             ["Epochs", f"max {P.EPOCHS}, early stopping (patience {P.PATIENCE}); run: " + ", ".join(f"{y}: {fo['epochs']}" for y, fo in folds.items())],
             ["Batch Size", P.BATCH], ["Optimizer", f"Adam (lr={P.LR})"],
             ["Loss Function", "Weighted sum: BCEWithLogits (clf) + MSE on standardized target (reg); weights " +
              ", ".join(f"{t}={P.TARGETS[t]['weight']}" for t in P.ACTIVE_TARGETS)],
             ["Validation", "Expanding walk-forward by year; validation = year before test; scaling fit on training only"]]), "",
         md(["Test year", "Train loss", "Val loss", "Seconds"],
            [[y, f(fo["train_loss"]), f(fo["val_loss"]), f"{fo['seconds']:.1f}"] for y, fo in folds.items()]), "",
         "## Step 4 – Deep Learning Evaluation", ""]
    for t in P.ACTIVE_TARGETS:
        keys = list(folds[years[0]]["metrics"][t])
        L += [f"**{t}**", "", md(["Metric"] + years + ["Mean"],
              [[k] + [f(folds[y]["metrics"][t][k]) for y in years] +
               [f(np.nanmean([folds[y]["metrics"][t][k] for y in years]))] for k in keys]), ""]
    L += ["Learning curves: `figures/learning_curves.png`; training/validation history: `history.json`.", "",
          "![learning curves](figures/learning_curves.png)", "",
          "## Step 5 – Benchmark Comparison", ""]
    # Compare DL vs ML baseline (XGBoost) on primary metric for each target, year. Compute mean gain and wins.
    if rows:
        L += [md(["Target", "Year", "DL", "ML baseline", "Gain"], [[t, y, f(d), f(b), f(g)] for t, y, d, b, g in rows]), ""]
    L += [md(["Criterion", "ML Model", "DL Model"], [
        ["Performance", "XGBoost, tabular (outputs_fase2)", f"LSTM; mean gain on primary metric {f(gain)}, wins {wins}/{n}"],
        ["Complexity", "Gradient-boosted trees, engineered lags/rolling features", f"{res['params']:,} parameters, {P.WINDOW}-day raw sequences"],
        ["Training Cost", f"{P.ML_TRAIN_SECONDS or 'not measured'} s", f"{secs:.0f} s total over {len(folds)} folds ({res['device']})"],
        ["Practical Value", "Operational pilot already defined (T2M_MIN, rain occurrence at h=1)", "Only valuable if it beats ML consistently; extensible via TARGETS registry"]]), "",
        "Note: baseline BSS uses the repo's smoothed climatology; DL BSS here uses a municipality × month climatology from training years.", "",
        "## Step 6 – Deep Learning Insights", ""]
    eces = [folds[y]["metrics"][t]["ece"] for y in years for t in P.ACTIVE_TARGETS if "ece" in folds[y]["metrics"][t]]
    L += [md(["Insight", "Evidence", "Practical Meaning"], [
        ["Improvement achieved", f"mean gain {f(gain)}, wins {wins}/{n} folds", "Positive and consistent gains are needed before replacing XGBoost" if n else "No baseline comparison available"],
        ["Limitations remain", f"{len(years)} folds, 2026 partial" + (f", mean ECE {f(float(np.mean(eces)))}" if eces else ""), "Confidence intervals not credible; check calibration before use"],
        ["Complexity justified?", f"{res['params']:,} params, {secs:.0f} s training", f"Verdict: {verdict}"]]), "",
        "## Step 7 – Innovation Assessment", "",
        md(["Criterion", "Assessment", "Evidence"], [
            ["Predictive Improvement", "Yes" if verdict.startswith("DEEP") else "No" if verdict.startswith("MACHINE") else "Inconclusive", f"wins {wins}/{n}, mean gain {f(gain)}"],
            ["Technical Feasibility", "Maintainable: single model class, config-driven targets", "7 small scripts, one extra dependency (torch)"],
            ["Computational Feasibility", "Reasonable on CPU for one-day horizon", f"{secs:.0f} s total, {res['infer_ms']:.2f} ms/1k sequences inference"],
            ["Potential Impact", "Shared encoder allows adding targets without new models", "Value depends on skill in more variables, which XGBoost audit did not find"]]), "",
        "## Step 8 – Deep Learning Recommendation", "", f"**{verdict}**", "",
        "### Executive Summary", "",
        f"- **Architecture:** multivariate LSTM, {P.WINDOW}-day window, {P.LSTM_LAYERS} layer(s), {P.HIDDEN} units, heads: {heads}.",
        f"- **Results:** " + "; ".join(f"{t}: " + ", ".join(f"{k}={f(np.nanmean([folds[y]['metrics'][t][k] for y in years]))}" for k in folds[years[0]]['metrics'][t]) for t in P.ACTIVE_TARGETS) + ".",
        f"- **Benchmark:** wins {wins}/{n} folds vs XGBoost, mean gain {f(gain)} (threshold {P.MIN_GAIN}).",
        "- **Main insights:** small dataset and 4 folds limit conclusions; calibration and target-leakage tests still pending per README.",
        f"- **Recommended approach:** {verdict.lower()}."]
    (P.OUT_DIR / "deep_learning.md").write_text("\n".join(L), encoding="utf-8")