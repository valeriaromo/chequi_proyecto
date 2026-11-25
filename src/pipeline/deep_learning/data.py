import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import params as P


def load_master():
    df = pd.read_parquet(P.MASTER_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["municipio", "date"]).reset_index(drop=True)


def add_targets(df, targets):
    for t in targets:
        spec = P.TARGETS[t]
        s = df.groupby("municipio", observed=True)[spec["col"]].shift(-P.HORIZON)
        df[f"y_{t}"] = (s >= spec["thr"]).astype(float).where(s.notna()) if spec["kind"] == "clf" else s
    return df


def feature_cols(df):
    return [c for c in df.select_dtypes("number").columns
            if c not in P.EXCLUDE_COLS and not c.startswith(P.EXCLUDE_PREFIXES)]


def ffill_features(df, feats):
    df[feats] = df.groupby("municipio", observed=True)[feats].ffill()  # past-only imputation
    return df


def sequences(df, feats, ycols):
    W, X, Y, D, M = P.WINDOW, [], [], [], []
    for _, g in df.groupby("municipio", observed=True, sort=False):
        a = g[feats].to_numpy(np.float32)
        X.append(sliding_window_view(a, W, axis=0).transpose(0, 2, 1))
        Y.append(g[ycols].to_numpy(np.float32)[W - 1:])
        D.append(g["date"].to_numpy()[W - 1:])
        M.append(g["municipio"].to_numpy()[W - 1:])
    X, Y, D, M = map(np.concatenate, (X, Y, D, M))
    ok = ~np.isnan(Y).any(axis=1)
    return X[ok], Y[ok], D[ok], M[ok]


def build_fold(df, feats, test_year, targets):
    """Scaler, target scale and climatology are fit on training years only."""
    fit = df[df.year < test_year - 1]
    mu, sd = fit[feats].mean(), fit[feats].std().replace(0, 1).fillna(1)
    s = df.copy()
    s[feats] = ((df[feats] - mu) / sd).fillna(0).astype(np.float32)

    yscale, clim = {}, {}
    for t in targets:
        c = f"y_{t}"
        f2 = fit.dropna(subset=[c])
        if P.TARGETS[t]["kind"] == "reg":
            yscale[t] = (f2[c].mean(), f2[c].std())
            s[c] = (df[c] - yscale[t][0]) / yscale[t][1]
        else:
            clim[t] = (f2.groupby(["municipio", f2.date.dt.month], observed=True)[c].mean(), f2[c].mean())

    X, Y, D, M = sequences(s, feats, [f"y_{t}" for t in targets])
    d, h = pd.DatetimeIndex(D), pd.Timedelta(days=P.HORIZON)
    tr = d < pd.Timestamp(test_year - 1, 1, 1) - h                     # purge: targets must not reach into next split
    va = (d >= pd.Timestamp(test_year - 1, 1, 1)) & (d < pd.Timestamp(test_year, 1, 1) - h)
    te = d.year == test_year
    return dict(Xtr=X[tr], Ytr=Y[tr], Xva=X[va], Yva=Y[va], Xte=X[te], Yte=Y[te],
                dates_te=D[te], mun_te=M[te], yscale=yscale, clim=clim)