import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
import cfg, common, data


def analyse(name):
    s = cfg.TARGETS[name]
    kind = s["kind"]
    df = data.load()
    Xtr, _, Xte, yte, meta = data.split(df, name)
    model = common.load_model(name)
    cols = list(Xte.columns)
    pred, y = common.predict(model, Xte, kind), yte.values
    shap_json = common.load_json(f"shap_{name}.json")

    med = Xtr.median()                                  # LIME cannot handle NaN -> impute with train median
    Xtr_i, Xte_i = Xtr.fillna(med).fillna(0), Xte.fillna(med).fillna(0)
    fn = (lambda a: model.predict_proba(pd.DataFrame(a, columns=cols))) if kind == "clf" \
        else (lambda a: model.predict(pd.DataFrame(a, columns=cols)))
    lime = LimeTabularExplainer(Xtr_i.values, feature_names=cols, mode="classification" if kind == "clf" else "regression",
                                class_names=["no_rain", "rain"], discretize_continuous=True, random_state=cfg.SEED)

    cases = common.select_cases(pred, y, kind)
    idx = [i for _, _, i in cases]
    sh = shap.TreeExplainer(model).shap_values(Xte.iloc[idx])
    sh = sh[:, :, 1] if np.ndim(sh) == 3 else sh

    rows, agg, ovs, sas = [], {}, [], []
    for k, (g, label, i) in enumerate(cases):
        exp = lime.explain_instance(Xte_i.iloc[i].values, fn, num_features=cfg.LIME_K, num_samples=cfg.LIME_N)
        pairs = [(cols[j], w) for j, w in exp.local_exp[1]]
        rules = [r for r, _ in exp.as_list(label=1)]
        rule_of = {f: r for (f, _), r in zip(pairs, rules)}
        top = sorted(pairs, key=lambda t: -abs(t[1]))
        for f, w in pairs:
            agg[f] = agg.get(f, 0) + abs(w)

        lime5 = {f: w for f, w in top[:5]}
        sh5 = {cols[j]: sh[k, j] for j in np.argsort(-np.abs(sh[k]))[:5]}
        both = set(lime5) & set(sh5)
        ov = len(both) / 5
        sa = float(np.mean([np.sign(lime5[f]) == np.sign(sh5[f]) for f in both])) if both else 0.0
        ovs.append(ov); sas.append(sa)
        coherent = ov >= 0.4 and sa >= 0.7

        ups = [f"{rule_of[f]}" for f, w in top if w > 0][:2]
        downs = [f"{rule_of[f]}" for f, w in top if w < 0][:2]
        key = "; ".join(f"{rule_of[f]} ({w:+.3f})" for f, w in top[:3])
        interp = (f"Raises: {', '.join(ups) or 'none'}. Lowers: {', '.join(downs) or 'none'}. "
                  f"Top-5 overlap with SHAP {ov:.0%}, sign agreement {sa:.0%} -> {'coherent' if coherent else 'weak agreement'}.")
        obs_id = f"{label} ({g}) | {meta['municipio'][i]} {meta['date'][i].date()} | pred={pred[i]:.2f} obs={y[i]:.2f}"
        rows.append(dict(observation=obs_id, key_factors=key, interpretation=interp, coherent=coherent,
                         group=g, overlap=ov, sign_agreement=sa))

        exp.as_pyplot_figure(label=1)
        import matplotlib.pyplot as plt
        plt.title(f"LIME - {name} - {label}")
        pth = common.save_fig(f"{name}_{label}_lime.png")
        common.add_viz(name, f"LIME - {label}", pth, f"Local linear surrogate around the {g} case {label}.",
                       f"{obs_id}. Key factors: {key}.",
                       "Gives a rule-level, non-technical reading of a single forecast; "
                       + ("confirms routine behaviour." if g == "representative" else "flags where forecasts should be questioned."))

    imp = pd.Series(shap_json["importance"])
    la = pd.Series(agg).reindex(imp.index).fillna(0)
    rc = spearmanr(la, imp)[0]
    common.save_json(dict(target=name, matrix=rows, mean_overlap=float(np.mean(ovs)),
                          mean_sign_agreement=float(np.mean(sas)), rank_corr_with_shap=None if np.isnan(rc) else float(rc),
                          coherent_cases=int(sum(r["coherent"] for r in rows)), n_cases=len(rows),
                          aggregate_importance=agg), f"lime_{name}.json")


def main():
    for name in cfg.TARGETS:
        analyse(name)


if __name__ == "__main__":
    main()