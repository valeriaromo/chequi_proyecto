import json
import numpy as np
import pandas as pd
import cfg
import common
from cfg import P
from deep_learning import data, evaluate


def check_grid(df):
    d = df.groupby("municipio", observed=True)["date"].diff().dropna()
    return int((d != pd.Timedelta(days=1)).sum())            # gaps would break shift(-h)


def check_alignment(df):
    # check that targets are aligned with features: y(t+h) = f(t) for each target, municipio and date
    out = {}
    for t, spec in P.TARGETS.items():
        fut = df[["municipio", "date", spec["col"]]].copy()
        fut["date"] -= pd.Timedelta(days=P.HORIZON)
        m = df[["municipio", "date", f"y_{t}"]].merge(fut, on=["municipio", "date"], how="left")
        v = m[spec["col"]]
        exp = (v >= spec["thr"]).astype(float).where(v.notna()) if spec["kind"] == "clf" else v
        both = m[f"y_{t}"].notna() & exp.notna()
        out[t] = dict(compared=int(both.sum()),mismatches=int((~np.isclose(m.loc[both, f"y_{t}"], exp[both])).sum()))
    return out


def placebo(df, feats):
    # shuffle targets per municipio, run a fold, check that DL does not beat the baseline
    rng = np.random.default_rng(cfg.SEEDS[0])
    d = df.copy()
    for t in P.TARGETS:
        c = f"y_{t}"
        d[c] = d.groupby("municipio", observed=True)[c].transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))

    # run a fold on the placebo data, check that  (BSS < cfg.PLACEBO_MAX for classification, R2 < cfg.PLACEBO_MAX_R2 for regression)
    r = common.run_fold(d, feats, cfg.PLACEBO_YEAR, list(P.TARGETS), cfg.SEEDS[0])
    m = evaluate.evaluate_fold(r["fold"], r["pred"], r["targets"])
    out, ok = {}, True

    for t in P.TARGETS:
        key, lim = ("bss", cfg.PLACEBO_MAX) if P.TARGETS[t]["kind"] == "clf" else ("r2", cfg.PLACEBO_MAX_R2)
        out[t] = {key: float(m[t][key]), "limit": lim}
        ok &= m[t][key] < lim
    out["passed"] = bool(ok)

    return out


def main():
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, feats = common.load()
    raw = data.load_master()
    res = dict(grid_gaps=check_grid(df), alignment=check_alignment(df),
               raw_nan_in_target_cols={t: int(raw[s["col"]].isna().sum()) for t, s in P.TARGETS.items()},
               split_boundary_rows=int(df.municipio.nunique() * P.HORIZON), placebo=placebo(df, feats))
    res["warnings"] = []
    if any(res["raw_nan_in_target_cols"].values()):
        res["warnings"].append("targets built after ffill: some targets are imputed values; build targets before ffill")
    res["warnings"].append(f"{res['split_boundary_rows']} rows per split boundary have targets in the next period "
                           "(h days); purge them in data.build_fold")
    res["passed"] = bool(res["grid_gaps"] == 0 and all(v["mismatches"] == 0 for v in res["alignment"].values())
                         and res["placebo"]["passed"])
    (cfg.OUT_DIR / "leakage.json").write_text(json.dumps(res, indent=2, default=float))
    print("leakage passed:", res["passed"])