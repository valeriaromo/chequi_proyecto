import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import cfg, common, data

PRACTICAL = {
    "representative": "Shows the model's routine reasoning; if it is sensible here, day-to-day forecasts are trustworthy.",
    "critical": "Shows why the model fails or hesitates; these are the situations where a user should not rely on it.",
}


def fmt(Xrow, V, idx):
    return "; ".join(f"{Xrow.index[j]}={Xrow.iloc[j]:.2f} ({V[j]:+.3f})" for j in idx)


def analyse(name):
    s = cfg.TARGETS[name]
    kind, unit = s["kind"], s["unit"]
    df = data.load()
    _, _, Xte, yte, meta = data.split(df, name)
    model = common.load_model(name)
    sv = shap.TreeExplainer(model)(Xte)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    V, base = sv.values, np.asarray(sv.base_values).reshape(-1)
    cols = list(Xte.columns)
    pred, y = common.predict(model, Xte, kind), yte.values

    raw = model.predict(Xte, output_margin=True)
    add_err = float(np.abs(base + V.sum(1) - raw).max())

    imp = pd.Series(np.abs(V).mean(0), index=cols).sort_values(ascending=False)
    share = imp / imp.sum()
    half = (meta["date"] < meta["date"].median()).values
    stab = spearmanr(np.abs(V[half]).mean(0), np.abs(V[~half]).mean(0))[0]
    err = (pred - y) ** 2 if kind == "clf" else np.abs(pred - y)
    ge = pd.Series(np.asarray(err)).groupby(meta["municipio"].values).mean()

    # ---- global: matrix + plots
    rows = []
    for f in imp.index[:cfg.TOP_K]:
        j = cols.index(f)
        rho = spearmanr(Xte[f], V[:, j], nan_policy="omit")[0] if Xte[f].nunique() > 1 else np.nan
        if np.isnan(rho) or abs(rho) < 0.15:
            txt = "weak or non-monotonic direction"
        else:
            txt = f"higher values {'increase' if rho > 0 else 'decrease'} {unit}"
        rows.append(dict(feature=f, shap_contribution=round(float(imp[f]), 4), share=round(float(share[f]), 3),
                         mean_signed=round(float(V[:, j].mean()), 4), rho=None if np.isnan(rho) else round(float(rho), 2),
                         positive_share=round(float((V[:, j] > 0).mean()), 2),
                         interpretation=f"{txt} (rho={rho:+.2f}); {share[f]:.0%} of total impact"))

    top3 = ", ".join(r["feature"] for r in rows[:3])
    shap.plots.beeswarm(sv, max_display=cfg.TOP_K, show=False)
    p1 = common.save_fig(f"{name}_shap_summary.png")
    common.add_viz(name, "SHAP summary (beeswarm)", p1,
                   f"How every feature value pushes each {len(Xte)} test predictions of {unit}.",
                   f"Top drivers: {top3}. Direction per driver: " +
                   "; ".join(f"{r['feature']} -> {r['interpretation'].split(' (')[0]}" for r in rows[:3]) + ".",
                   "Reveals overall model behaviour and whether drivers match meteorological common sense.")
    shap.plots.bar(sv, max_display=cfg.TOP_K, show=False)
    p2 = common.save_fig(f"{name}_shap_importance.png")
    common.add_viz(name, "SHAP global importance (bar)", p2, "Mean absolute SHAP value per feature.",
                   f"Top-5 features carry {share.iloc[:5].sum():.0%} of total impact; top-1 ({imp.index[0]}) {share.iloc[0]:.0%}.",
                   "Tells stakeholders which inputs to monitor first and which data failures would hurt most.")

    # ---- local
    local = []
    for g, label, i in common.select_cases(pred, y, kind):
        order = np.argsort(V[i])
        neg = [j for j in order[:3] if V[i, j] < 0]
        pos = [j for j in order[::-1][:3] if V[i, j] > 0]
        r = Xte.iloc[i]
        tag = f"{name}_{label}"
        shap.plots.waterfall(sv[i], max_display=10, show=False)
        pw = common.save_fig(f"{tag}_waterfall.png")
        shap.force_plot(float(base[i]), V[i], r.round(2), matplotlib=True, show=False)
        pf = common.save_fig(f"{tag}_force.png")
        head = f"{meta['municipio'][i]} {meta['date'][i].date()}: predicted {pred[i]:.2f}, observed {y[i]:.2f}."
        obs = f"{head} Pushes up: {fmt(r, V[i], pos) or 'none'}. Pushes down: {fmt(r, V[i], neg) or 'none'}."
        common.add_viz(name, f"SHAP waterfall - {label}", pw, f"Additive breakdown of one {g} prediction ({label}).", obs, PRACTICAL[g])
        common.add_viz(name, f"SHAP force - {label}", pf, f"Push/pull of features around the base value for the {label} case.", obs, PRACTICAL[g])
        local.append(dict(id=label, group=g, pos=i, date=meta["date"][i], municipio=meta["municipio"][i],
                          pred=float(pred[i]), actual=float(y[i]), top_positive=fmt(r, V[i], pos),
                          top_negative=fmt(r, V[i], neg)))

    common.save_json(dict(
        target=name, kind=kind, unit=unit, n_test=len(Xte), base_value=float(base.mean()),
        additivity_err=add_err, stability_rho=float(stab), top1_share=float(share.iloc[0]),
        top5_mass=float(share.iloc[:5].sum()), top10_mass=float(share.iloc[:10].sum()),
        importance={k: float(v) for k, v in imp.items()}, matrix=rows, local=local,
        group_error={k: float(v) for k, v in ge.items()}, bias_ratio=float(ge.max() / max(ge.min(), 1e-12))),
        f"shap_{name}.json")


def main():
    for name in cfg.TARGETS:
        analyse(name)


if __name__ == "__main__":
    main()