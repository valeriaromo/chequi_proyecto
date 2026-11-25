import pandas as pd
import cfg

def load():
    df = pd.read_parquet(cfg.MASTER)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["municipio", "date"]).reset_index(drop=True)
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


def split(df, name):
    cols, y, d = features(df), df["y_" + name], df["date"]
    start = pd.Timestamp(year=cfg.TEST_YEAR, month=1, day=1)
    ok = y.notna()
    tr = ok & (d < start - pd.Timedelta(days=cfg.H))      # purge: targets of last H days leak into test
    te = ok & (d.dt.year == cfg.TEST_YEAR)
    pick = lambda m: (df.loc[m, cols].reset_index(drop=True), y[m].reset_index(drop=True))
    Xtr, ytr = pick(tr)
    Xte, yte = pick(te)
    meta = df.loc[te, ["date", "municipio"]].reset_index(drop=True)
    return Xtr, ytr, Xte, yte, meta