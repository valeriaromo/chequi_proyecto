import json
import numpy as np
import pandas as pd
import cfg, common
from deep_learning import params as P

# primary metric per target type: BSS for classification, skill (R2) for regression
PRIM = {"clf": "bss", "reg": "skill"}


def main():
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, feats = common.load()
    rows, cal = [], {}

    # run every config, seed, year, target; collect per fold metrics and calibration data
    for cname, targets in cfg.CONFIGS.items():
        for seed in cfg.SEEDS:
            for year in P.TEST_YEARS:
                r = common.run_fold(df, feats, year, targets, seed)
                for t in targets:
                    rows.append(dict(config=cname, target=t, seed=seed, year=year, secs=r["secs"],
                                     epochs=len(r["hist"]["val_loss"]), **common.score(t, r, df, year)))
                    if seed == cfg.SEEDS[0] and P.TARGETS[t]["kind"] == "clf":
                        i = r["targets"].index(t)
                        # collect (y_true, y_pred) for calibration plots; one per config + target
                        cal.setdefault((cname, t), []).append((r["fold"]["Yte"][:, i], r["pred"][:, i]))
                print(cname, seed, year, flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(cfg.OUT_DIR / "benchmark_runs.csv", index=False)

    summary = {}
    for (cname, t), g in runs.groupby(["config", "target"], observed=True):
        m = PRIM[P.TARGETS[t]["kind"]]

        #
        entry = dict(metric=m, per_year={str(y): float(v) for y, v in g.groupby("year", observed=True)[m].mean().items()},
                     mean=float(g.groupby("seed", observed=True)[m].mean().mean()), seed_std=float(g.groupby("seed", observed=True)[m].mean().std()),
                     secs_per_fold=float(g["secs"].mean()))
        if P.TARGETS[t]["kind"] == "clf":
            entry.update({k: float(g[k].mean()) for k in ("auc", "ece", "precision", "recall", "f1", "thr")})
        summary.setdefault(t, {})[cname] = entry
    (cfg.OUT_DIR / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    tables = []
    for (cname, t), parts in cal.items():
        tab = common.reliability(np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts]))
        tab.insert(0, "config", cname)
        tab.insert(1, "target", t)
        tables.append(tab)
    if tables:
        pd.concat(tables).to_csv(cfg.OUT_DIR / "calibration.csv", index=False)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for (cname, t), tab in pd.concat(tables).groupby(["config", "target"], observed=True):
            ax.plot(tab["mean_pred"], tab["observed"], marker="o", label=f"{cname}|{t}")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(cfg.FIG_DIR / "calibration.png", dpi=120)