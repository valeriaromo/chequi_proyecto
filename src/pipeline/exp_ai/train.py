import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import cfg, common, data
import pandas as pd

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


def main():
    common.make_dirs()
    df = data.load()
    metrics = {}
    for name, s in cfg.TARGETS.items():
        Xtr, ytr, Xte, yte, meta = data.split(df, name)
        model = (XGBClassifier if s["kind"] == "clf" else XGBRegressor)(**cfg.PARAMS).fit(Xtr, ytr)
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
        m.update(n_train=len(Xtr), n_test=len(Xte), n_features=Xte.shape[1], test_year=cfg.TEST_YEAR)
        metrics[name] = m
    common.save_json(metrics, "metrics.json")


if __name__ == "__main__":
    main()