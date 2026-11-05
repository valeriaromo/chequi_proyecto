from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import test_pca as T
from variable_analysis import default_master_path, load_data
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))
from pipeline import train_xgboost as P

EXCLUDE = set(P.EXCLUDE) | {"date","municipio","year",}
AOD_PATTERNS = ("aod","qc_aod",)

def is_aod(col: str) -> bool:
    name = col.lower()
    return any(pattern in name for pattern in AOD_PATTERNS)

def get_features(master_cols: list[str]) -> list[str]:
    features = [
        c for c in master_cols if c not in EXCLUDE and not P.is_aod(c)and not is_aod(c)
        ]
    return features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    path = Path(a.data) if a.data else default_master_path()
    master_cols = pq.read_schema(path).names
    feats = get_features(master_cols)
    print(f"Variables totales en master: {len(master_cols)}")
    print(f"Variables excluidas: {len(master_cols) - len(feats)}")
    print(f"Features finales sin AOD: {len(feats)}")
    feats = get_features(master_cols)
    refs = {"ref_master_sin_aod": feats,}
    print("Features seleccionadas:")
    print(f"Total: {len(feats)}")
    print("\nFeatures:")

    for c in feats:
        print(f" - {c}")

    df, _ = load_data(path)
    df["year"] = df["date"].dt.year
    df[T.RAIN_COL + "_raw"] = df[T.RAIN_COL]
    rows = []

    for yr in T.TEST_YEARS:
        _, _, y_t, y_r, p_clim, train, test, _ = T.fold_data(df,[T.TARGET],yr)
        skill, bss = T.evaluate(df[feats],y_t,y_r,p_clim,train,test)
        rows.append({"ref": "ref_master_sin_aod","n_feats": len(feats),"test_year": yr,"skill_T2M_MIN": skill,"BSS_rain": bss,})

    res = pd.DataFrame(rows)
    out = (Path(a.out) if a.out else T.OUT_DIR) / "ref_produccion"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "ref_folds.csv", index=False)
    print("\nResultados por año:")
    print(res.round(3))
    print("\nResumen:")
    print(res.groupby("ref")[["skill_T2M_MIN", "BSS_rain"]].agg(["mean", "min"]).round(3))

if __name__ == "__main__":
    main()