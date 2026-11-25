from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import cfg, common, data


def smooth_clim(df, name, meta):
    cut = pd.Timestamp(year=cfg.TEST_YEAR, month=1, day=1) - pd.Timedelta(days=cfg.H)
    tr = df[(df["date"] < cut) & df["y_" + name].notna()]
    tab = (tr.assign(doy=tr["date"].dt.dayofyear)
             .pivot_table(index="municipio", columns="doy", values="y_" + name)
             .reindex(columns=range(1, 367)).interpolate(axis=1, limit_direction="both"))
    w = 7
    ext = pd.concat([tab.iloc[:, -w:], tab, tab.iloc[:, :w]], axis=1)
    sm = ext.T.rolling(2 * w + 1, center=True).mean().T.iloc[:, w:-w]
    r = sm.index.get_indexer(meta["municipio"])
    c = meta["date"].dt.dayofyear.values - 1
    return sm.to_numpy()[r, c].clip(0.01, 0.99)


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(y))


def tune(df, name, kind):
    """Walk-forward grid search on years < TEST_YEAR only (Brier for rain, MAE for temperature)."""
    Model = XGBClassifier if kind == "clf" else XGBRegressor
    best, best_score = None, np.inf
    for d, mcw, (lr, n) in product(cfg.GRID["max_depth"], cfg.GRID["min_child_weight"], cfg.GRID["lr_n"]):
        p = {**cfg.PARAMS, "max_depth": d, "min_child_weight": mcw, "learning_rate": lr, "n_estimators": n}
        scores = []
        for yr in cfg.CV_YEARS:
            Xa, ya, Xv, yv, _ = data.split(df, name, yr)
            pr = common.predict(Model(**p).fit(Xa, ya), Xv, kind)
            scores.append(brier_score_loss(yv, pr) if kind == "clf" else mean_absolute_error(yv, pr))
        score = float(np.mean(scores))
        if score < best_score:
            best, best_score = p, score
    return best, best_score


def main():
    common.make_dirs()
    df = data.load()
    metrics = {}
    for name, s in cfg.TARGETS.items():
        Xtr, ytr, Xte, yte, meta = data.split(df, name)
        params, cv = tune(df, name, s["kind"])
        model = (XGBClassifier if s["kind"] == "clf" else XGBRegressor)(**params).fit(Xtr, ytr)
        joblib.dump(model, cfg.MODELS / f"{name}.joblib")
        p, y = common.predict(model, Xte, s["kind"]), yte.values
        if s["kind"] == "clf":
            brier = brier_score_loss(y, p)
            clim = brier_score_loss(y, smooth_clim(df, name, meta))
            m = dict(auc=roc_auc_score(y, p), brier=brier, bss=1 - brier / clim, ece=ece(y, p))
        else:
            mae = mean_absolute_error(y, p)
            m = dict(mae=mae, r2=r2_score(y, p))
            if s["src"] in Xte:                                   # persistence baseline
                m["skill_vs_persistence"] = 1 - mae / mean_absolute_error(y, Xte[s["src"]].fillna(ytr.mean()))
        m.update(cv_score=cv, n_train=len(Xtr), n_test=len(Xte), n_features=Xte.shape[1], test_year=cfg.TEST_YEAR,
                 tuned_params=(f"depth={params['max_depth']}, mcw={params['min_child_weight']}, "
                               f"lr={params['learning_rate']}, n={params['n_estimators']}"))
        metrics[name] = m
    common.save_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()