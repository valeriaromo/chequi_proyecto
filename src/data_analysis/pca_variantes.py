from __future__ import annotations
import numpy as np, pandas as pd
import test_pca as T
from pca_bloques import BLOCKS, STATIC, CAL, fit_block, master_features

LEVELS = (0.80, 0.90)     # varianza acumulada mínima por bloque, calculada en train de cada fold
RHO = 0.95                # |corr| arriba de esto = redundante (variante podada)


def prune(df, train, cols, rho=RHO):
    # poda voraz con correlación solo de train: conserva la primera de cada par redundante
    C = df.loc[train, cols].corr().abs()
    keep: list[str] = []
    for c in cols:
        if all(C.loc[c, k] <= rho for k in keep):
            keep.append(c)
    return keep


def main() -> None:
    path = T.default_master_path()
    feats = master_features(path)
    df, _ = T.load_data(path)
    df["year"] = df["date"].dt.year
    df[T.RAIN_COL + "_raw"] = df[T.RAIN_COL]
    elig = sum(BLOCKS.values(), [])
    fixed = df[STATIC + CAL]
    rows = []

    for yr in T.TEST_YEARS:
        _, _, y_t, y_r, p_clim, train, test, _ = T.fold_data(df, [T.TARGET], yr)
        fits = {b: fit_block(df, c, train) for b, c in BLOCKS.items()}   # climatología, z-score y PCA solo con train
        keep = prune(df, train, elig)
        podado = [c for c in feats if c not in elig or c in keep]

        def add(name, level, X):
            skill, bss = T.evaluate(X, y_t, y_r, p_clim, train, test)
            rows.append(dict(variante=name, nivel=level, fold=yr, n_feats=X.shape[1], skill=skill, bss=bss))

        add("orig65", "-", df[feats])
        for lv in LEVELS:
            ks = {b: int(np.searchsorted(cum, lv) + 1) for b, (_, cum) in fits.items()}
            P = pd.concat([s.iloc[:, :ks[b]].add_prefix(f"{b}_") for b, (s, _) in fits.items()], axis=1)
            PL = pd.concat([P.groupby(df["municipio"]).shift(l).add_suffix(f"_t{l}") for l in (1, 2)], axis=1)
            add("solo_pcs", lv, pd.concat([P, fixed], axis=1))          # PCs + estáticas + calendario
            add("orig+pcs", lv, pd.concat([df[feats], P], axis=1))
            add("podado+pcs", lv, pd.concat([df[podado], P], axis=1))
            add("pcs+pcs_lag", lv, pd.concat([P, PL, fixed], axis=1))   # PCs en t, t-1, t-2
        print(f"fold {yr}: k por bloque {ks} (nivel {LEVELS[-1]}), podadas {len(elig) - len(keep)}")

    res = pd.DataFrame(rows)
    out = T.OUT_DIR / "pca_variantes"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "folds.csv", index=False)

    ref = res[res.variante == "orig65"].set_index("fold")
    summ = [dict(variante="orig65", nivel="-", n_feats=int(ref.n_feats.mean()), skill=ref.skill.mean(),
                 skill_min=ref.skill.min(), bss=ref.bss.mean(), bss_min=ref.bss.min(), cumple="ref")]
    for (v, lv), g in res[res.variante != "orig65"].groupby(["variante", "nivel"], sort=False):
        g = g.set_index("fold")
        ok = (g.skill.mean() >= ref.skill.mean() - T.TOL_MEAN and g.bss.mean() >= ref.bss.mean() - T.TOL_MEAN
              and (g.skill >= ref.skill - T.TOL_FOLD).all() and (g.bss >= ref.bss - T.TOL_FOLD).all())
        summ.append(dict(variante=v, nivel=lv, n_feats=int(round(g.n_feats.mean())), skill=g.skill.mean(),
                         skill_min=g.skill.min(), bss=g.bss.mean(), bss_min=g.bss.min(), cumple=bool(ok)))
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "resumen.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()