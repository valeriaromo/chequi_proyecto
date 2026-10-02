import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.settings import (HORIZONTES,MARGEN_AOD,MASTER_TABLE_PATH,MIN_TRAIN_YEARS,REPORTS_DIR,SEED,TARGETS,)

EXCLUDE = {"date", "municipio", "year", "qc_aod_imputed"}
PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=5,subsample=0.8, colsample_bytree=0.8, tree_method="hist")

# se usarán 2 corridas para comparar el uso de la base de datos 
#  si conviene usar AOD como feature: A sin AOD, B con AOD

def is_aod(col):
    return col.startswith("aod_") or col == "qc_aod_pocos_pixeles"

def objective(target):
    if target == "precipitation_mm":
        return dict(objective="reg:tweedie", tweedie_variance_power=1.5)
    return dict(objective="reg:squarederror")

def metrics(y, pred):
    err = y.to_numpy() - np.asarray(pred)
    return float(np.nanmean(np.abs(err))), float(np.sqrt(np.nanmean(err**2)))


def climatology(tr, te):
    # se groupea por municipio y doy_t (día del año + horizonte) para calcular la predicción climatológica promedio de la variable objetivo
    clim = tr.groupby(["municipio", "doy_t"])["y"].mean().rename("clim").reset_index()
    pred = te[["municipio", "doy_t"]].merge(clim, on=["municipio", "doy_t"], how="left")["clim"]
    fallback = te["municipio"].map(tr.groupby("municipio")["y"].mean())
    return pred.fillna(fallback.reset_index(drop=True)).to_numpy()


def main():
    df = pd.read_parquet(MASTER_TABLE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["municipio", "date"]).reset_index(drop=True)

    feats_b = [c for c in df.columns if c not in EXCLUDE]
    feats_a = [c for c in feats_b if not is_aod(c)]
    runs = {"xgb_A": feats_a, "xgb_B": feats_b}
    years = sorted(df["year"].unique())[MIN_TRAIN_YEARS:]
    # se usa walk-forward por year: cada año es test y los anteriores son train
    # purgando h días antes del test para evitar que la variable objetivo (y_t+h) toque el test

    rows = []
    for tgt in TARGETS:
        for h in HORIZONTES:
            d = df.assign(y=df.groupby("municipio")[tgt].shift(-h)).dropna(subset=["y"])
            d = d.assign(doy_t=(d["date"] + pd.Timedelta(days=h)).dt.dayofyear)
            for yr in years:
                start = pd.Timestamp(year=int(yr), month=1, day=1)
                tr = d[d["date"] < start - pd.Timedelta(days=h)]  # purga: y_t+h no toca el test
                te = d[d["year"] == yr]
                if tr.empty or te.empty:
                    continue

                # compara con persistencia y climatología para ver si puede mejorar el MAE y si conviene usar AOD como feature
                preds = {"persistencia": te[tgt].to_numpy(), "climatologia": climatology(tr, te)}
                for name, feats in runs.items():
                    model = xgb.XGBRegressor(**PARAMS, **objective(tgt), random_state=SEED, n_jobs=-1)
                    model.fit(tr[feats], tr["y"])
                    preds[name] = model.predict(te[feats])

                for name, p in preds.items():
                    mae, rmse = metrics(te["y"], p)
                    rows.append(dict(target=tgt, h=h, year=yr, model=name, mae=mae, rmse=rmse))
                print(f"{tgt} h={h} year={yr} ok")

    res = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS_DIR / "xgb_folds.csv", index=False)

    folds = res.pivot_table(index=["target", "h", "year"], columns="model", values="mae")
    mean = folds.groupby(level=["target", "h"]).mean()
    win_b = (folds["xgb_B"] < folds["xgb_A"]).groupby(level=["target", "h"]).mean()

    out = mean.copy()
    out["mejor_baseline"] = mean[["persistencia", "climatologia"]].min(axis=1)
    out["skill_A"] = 1 - mean["xgb_A"] / out["mejor_baseline"]
    out["skill_B"] = 1 - mean["xgb_B"] / out["mejor_baseline"]
    out["folds_B_gana"] = win_b
    out["usar_xgb"] = mean[["xgb_A", "xgb_B"]].min(axis=1) < out["mejor_baseline"]
    out["usar_aod"] = (mean["xgb_B"] < mean["xgb_A"] * (1 - MARGEN_AOD)) & (win_b > 0.5)

    out.to_csv(REPORTS_DIR / "xgb_decision.csv")
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()