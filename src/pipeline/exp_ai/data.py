import numpy as np
import pandas as pd
import cfg


def _engineered(df):
    """Extra predictors built only from day-t and earlier information."""
    g = df.groupby("municipio")
    new = {}
    for col, tag in (("precipitation_mm", "precip"), ("PRECTOTCORR", "prectot")):
        if col in df.columns:                                   # same-day, all municipalities
            new[f"dom_{tag}_mean"] = df.groupby("date")[col].transform("mean")
            wet = (df[col] >= cfg.RAIN_MM).astype(float).where(df[col].notna())
            new[f"dom_{tag}_wet"] = wet.groupby(df["date"]).transform("mean")
    for col, k in (("PS", 1), ("RH2M", 1), ("T2M_MIN", 1), ("T2M_MIN", 3)):
        if col in df.columns:
            new[f"d{k}_{col}"] = g[col].diff(k)
    if {"T2M_MIN", "T2M_mean30d"} <= set(df.columns):
        new["T2M_MIN_anom30d"] = df["T2M_MIN"] - df["T2M_mean30d"]
    if {"T2M", "RH2M"} <= set(df.columns):                      # Magnus dewpoint, T2M in °C
        a, b = 17.62, 243.12
        gam = np.log(df["RH2M"].clip(1, 100) / 100) + a * df["T2M"] / (b + df["T2M"])
        td = b * gam / (a - gam)
        new["dewpt"], new["dew_dep"] = td, df["T2M"] - td
    return pd.concat([df, pd.DataFrame(new)], axis=1)


def load():
    df = pd.read_parquet(cfg.MASTER)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["municipio", "date"]).reset_index(drop=True)
    df = _engineered(df)
    g = df.groupby("municipio")
    for name, s in cfg.TARGETS.items():
        assert s["src"] in df.columns, f"cfg.TARGETS['{name}']['src']='{s['src']}' not in master columns"
        nxt = g[s["src"]].shift(-cfg.H)
        y = (nxt >= cfg.RAIN_MM).astype(float) if s["kind"] == "clf" else nxt
        df["y_" + name] = y.where(nxt.notna())
    return df


def features(df):
    drop = {"date", "municipio", "year", "qc_aod_imputed", *cfg.EXTRA_DROP}
    return [c for c in df.columns
            if c not in drop and not c.startswith(("y_", "aod", "qc_"))
            and pd.api.types.is_numeric_dtype(df[c])]


def split(df, name, year=None):
    """Train on dates < year (purged by H), evaluate on `year`. Default year = cfg.TEST_YEAR."""
    year = year or cfg.TEST_YEAR
    cols, y, d = features(df), df["y_" + name], df["date"]
    start = pd.Timestamp(year=year, month=1, day=1)
    ok = y.notna()
    tr = ok & (d < start - pd.Timedelta(days=cfg.H))      # purge: targets of last H days leak into test
    te = ok & (d.dt.year == year)
    pick = lambda m: (df.loc[m, cols].reset_index(drop=True), y[m].reset_index(drop=True))
    Xtr, ytr = pick(tr)
    Xte, yte = pick(te)
    meta = df.loc[te, ["date", "municipio"]].reset_index(drop=True)
    return Xtr, ytr, Xte, yte, meta