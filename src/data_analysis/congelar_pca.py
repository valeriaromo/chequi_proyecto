from __future__ import annotations
import argparse, hashlib, json, platform
from datetime import date
from pathlib import Path
import joblib, numpy as np, pandas as pd
import sklearn, xgboost
from xgboost import XGBClassifier, XGBRegressor
import pca_analysis as pca
import test_pca as T
from pca_bloques import BLOCKS, LOG1P, STATIC, CAL, check_cols, fit_block, master_features

# En produccion, cada dia nuevo se transforma con estos mismos objetos (transform_block / predict_dia) nada se reajusta.

# hiperparametros validados en el walk forward (mantener iguales a test_pca.XGB y a pca_regularizado.py)
PARAMS = {
    "base": dict(T.XGB),
    "reg": {**T.XGB, "n_estimators": 150, "max_depth": 3, "learning_rate": 0.03,
            "min_child_weight": 50, "reg_lambda": 10.0},
}

# decision por target y horizonte (rep=None -> sin modelo util: se predice la climatologia)
CONFIG = {
    ("T2M_MAX", 1): dict(rep="orig+pcs", level=0.90, params="base", wf_skill=0.455, nota="PCs como features adicionales"),
    ("T2M_MIN", 1): dict(rep="orig+pcs", level=0.90, params="base", wf_skill=0.467, nota="PCs como features adicionales"),
    ("T2M_MAX", 3): dict(rep="orig+pcs", level=0.95, params="reg", wf_skill=0.141, nota="solo_pcs 0.95 mejora mas pero falla la tolerancia por fold por 0.001"),
    ("T2M_MIN", 3): dict(rep="solo_pcs", level=0.95, params="reg", wf_skill=0.077, nota="PCs reemplazan al master (45 features)"),
    ("T2M_MAX", 7): dict(rep="orig65", level=None, params="reg", wf_skill=0.051, nota="marginal: los PCs no aportan; un fold bajo climatologia"),
    ("T2M_MIN", 7): dict(rep=None, level=None, params=None, wf_skill=-0.040, nota="skill negativo en todas las representaciones"),
    ("lluvia", 1): dict(rep="orig65", level=None, params="base", wf_skill=0.229, nota="el PCA no ayuda a lluvia"),
    ("lluvia", 3): dict(rep=None, level=None, params=None, wf_skill=-0.026, nota="BSS negativo (falta probar con parametros regularizados)"),
    ("lluvia", 7): dict(rep=None, level=None, params=None, wf_skill=-0.036, nota="BSS negativo (falta probar con parametros regularizados)"),
}
TEMP_TARGETS = ("T2M_MAX", "T2M_MIN")


# climatologia congelada (misma formula que test_pca.anomalies, pero guardando el perfil)
def clim_profile(F: pd.DataFrame, col: str, mask, munis: pd.Index) -> np.ndarray:
    tr = F[mask]
    prof = (tr.groupby(["municipio", tr["date"].dt.dayofyear.clip(upper=366)])[col].mean().unstack()
            .reindex(index=munis, columns=range(1, 367)))
    ext = pd.concat([prof.iloc[:, -T.WIN:], prof, prof.iloc[:, :T.WIN]], axis=1)
    return ext.T.rolling(T.WIN, center=True, min_periods=1).mean().T.iloc[:, T.WIN:-T.WIN].to_numpy()


def clim_value(F: pd.DataFrame, prof: np.ndarray, munis: pd.Index, shift_days: int = 0) -> np.ndarray:
    doy = (F["date"] + pd.Timedelta(days=shift_days)).dt.dayofyear.clip(upper=366).to_numpy()
    mrow = munis.get_indexer(F["municipio"])
    assert (mrow >= 0).all(), "municipio desconocido: el PCA congelado no lo conoce"
    return prof[mrow, doy - 1]


def prep_block(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    X = df[cols].astype(float)
    for c in cols:
        if c in LOG1P:
            X[c] = np.log1p(X[c].clip(lower=0))
    return df[["municipio", "date"]].join(X)


def freeze_block(df, cols, mask) -> dict:
    munis = pd.Index(sorted(df["municipio"].unique()))
    F = prep_block(df, cols)
    profs = {c: clim_profile(F, c, mask, munis) for c in cols}
    A = pd.DataFrame({c: F[c].to_numpy() - clim_value(F, profs[c], munis) for c in cols}, index=df.index)
    mu, sd = A[mask].mean(), A[mask].std(ddof=0).replace(0, 1.0)
    Z = ((A - mu) / sd).fillna(0.0)
    res = pca.fit_pca(Z[mask])
    return dict(cols=list(cols), munis=list(munis), profs=profs, mu=mu, sd=sd,
                W=res.loadings.to_numpy() / np.sqrt(res.eigenvalues), names=list(res.loadings.columns),
                eigenvalues=np.asarray(res.eigenvalues), cum=np.cumsum(res.eigenvalues) / np.sum(res.eigenvalues))


def transform_block(df: pd.DataFrame, b: dict) -> pd.DataFrame:
    munis = pd.Index(b["munis"])
    F = prep_block(df, b["cols"])
    A = pd.DataFrame({c: F[c].to_numpy() - clim_value(F, b["profs"][c], munis) for c in b["cols"]}, index=df.index)
    Z = ((A - b["mu"]) / b["sd"]).fillna(0.0)
    return pd.DataFrame(Z.to_numpy() @ b["W"], index=df.index, columns=b["names"])


def k_for(b: dict, level: float) -> int:
    return int(np.searchsorted(b["cum"], level) + 1)


def pcs(df: pd.DataFrame, art: dict, level: float) -> pd.DataFrame:
    parts = [transform_block(df, art["blocks"][n]).iloc[:, :k_for(art["blocks"][n], level)].add_prefix(f"{n}_")
             for n in BLOCKS]
    return pd.concat(parts, axis=1)


def build_X(df, art, feats, rep, level) -> pd.DataFrame:
    if rep == "orig65":
        return df[feats]
    P = pcs(df, art, level)
    return pd.concat([P, df[STATIC + CAL]], axis=1) if rep == "solo_pcs" else pd.concat([df[feats], P], axis=1)


# congelado y entrenamiento
def freeze_all(df: pd.DataFrame, feats: list[str]) -> dict:
    mask = pd.Series(True, index=df.index)            # todos los años disponibles
    munis = pd.Index(sorted(df["municipio"].unique()))
    F = df[["municipio", "date", *TEMP_TARGETS]]
    art = dict(blocks={n: freeze_block(df, c, mask) for n, c in BLOCKS.items()},
               target_clim={t: clim_profile(F, t, mask, munis) for t in TEMP_TARGETS},
               munis=list(munis), feats=list(feats), p_clim={}, fecha_ini=str(df["date"].min().date()),
               fecha_fin=str(df["date"].max().date()))
    for h in sorted({h for (t, h) in CONFIG if t == "lluvia"}):
        w = rain_label(df, h)
        tm = (df["date"] + pd.Timedelta(days=h)).dt.month
        art["p_clim"][h] = dict(tabla=pd.DataFrame({"m": df["municipio"], "mo": tm, "y": w}).dropna()
                                .groupby(["m", "mo"])["y"].mean(), global_=float(w.mean()))
    return art


def rain_label(df, h) -> pd.Series:
    raw = df.groupby("municipio")[T.RAIN_COL].shift(-h)
    return (raw >= T.RAIN_MM).astype(float).where(raw.notna())


def train_models(df, art, feats) -> dict:
    munis = pd.Index(art["munis"])
    models = {}
    for (tgt, h), c in CONFIG.items():
        if c["rep"] is None:
            continue
        X = build_X(df, art, feats, c["rep"], c["level"])
        if tgt == "lluvia":
            y = rain_label(df, h)
            m = y.notna()
            model = XGBClassifier(**PARAMS[c["params"]], eval_metric="logloss").fit(X[m], y[m])
        else:
            anom = df[tgt].to_numpy() - clim_value(df, art["target_clim"][tgt], munis)
            y = pd.Series(anom, index=df.index).groupby(df["municipio"]).shift(-h)
            m = y.notna()
            model = XGBRegressor(**PARAMS[c["params"]]).fit(X[m], y[m])
        models[(tgt, h)] = dict(model=model, columns=list(X.columns), kind="clf" if tgt == "lluvia" else "reg", **c)
        print(f"modelo {tgt} h={h}: {c['rep']} {c['level'] or ''} ({X.shape[1]} features, {int(m.sum())} filas)")
    return models


def predict_dia(df_rows: pd.DataFrame, tgt: str, h: int, art: dict, models: dict) -> pd.DataFrame:
    # Prediccion para filas nuevas del master (features ya construidas) sin reajustar nada
    munis = pd.Index(art["munis"])
    out = df_rows[["municipio", "date"]].copy()
    out["fecha_objetivo"] = out["date"] + pd.Timedelta(days=h)
    mdl = models.get((tgt, h))
    if tgt == "lluvia":
        pc = art["p_clim"][h]
        base = pd.Series(pc["tabla"].reindex(pd.MultiIndex.from_arrays(
            [df_rows["municipio"], out["fecha_objetivo"].dt.month])).to_numpy(), index=df_rows.index).fillna(pc["global_"])
        out["prob_lluvia"] = base if mdl is None else mdl["model"].predict_proba(
            build_X(df_rows, art, art["feats"], mdl["rep"], mdl["level"])[mdl["columns"]])[:, 1]
    else:
        clim = clim_value(df_rows, art["target_clim"][tgt], munis, shift_days=h)
        out[f"{tgt}_pred"] = clim if mdl is None else clim + mdl["model"].predict(
            build_X(df_rows, art, art["feats"], mdl["rep"], mdl["level"])[mdl["columns"]])
    out["fuente"] = "climatologia" if mdl is None else f"{mdl['rep']}"
    return out


# guardado y verificacion 
def save_all(art, models, out_dir: Path, master_path: Path, version: str) -> None:
    (out_dir / "modelos").mkdir(parents=True, exist_ok=True)
    joblib.dump(art, out_dir / "pca_bloques.joblib")
    for (tgt, h), m in models.items():
        joblib.dump(m, out_dir / "modelos" / f"{tgt}_h{h}.joblib")
    manifest = dict(
        version=version, creado=str(date.today()), master=master_path.name,
        master_sha256=hashlib.sha256(master_path.read_bytes()).hexdigest(),
        fecha_ini=art["fecha_ini"], fecha_fin=art["fecha_fin"], n_feats_master=len(art["feats"]),
        k_por_nivel={str(lv): {n: k_for(b, lv) for n, b in art["blocks"].items()} for lv in (0.80, 0.90, 0.95)},
        varianza_acumulada_bloques={n: [round(float(x), 4) for x in b["cum"]] for n, b in art["blocks"].items()},
        config={f"{t}_h{h}": c for (t, h), c in CONFIG.items()},
        params=PARAMS, libs=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                                 sklearn=sklearn.__version__, xgboost=xgboost.__version__))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))


def verify(df, feats, out_dir: Path) -> None:
    art = joblib.load(out_dir / "pca_bloques.joblib")             # lo que leeria produccion
    models = {(p.stem.rsplit("_h", 1)[0], int(p.stem.rsplit("_h", 1)[1])): joblib.load(p)
              for p in (out_dir / "modelos").glob("*.joblib")}
    mask = pd.Series(True, index=df.index)

    # 1 transformar con el objeto congelado, ajustar y transformar en el mismo conjunto (fit_block)
    for n, cols in BLOCKS.items():
        s_ref, _ = fit_block(df, cols, mask)
        s_new = transform_block(df, art["blocks"][n])
        assert np.allclose(s_ref.to_numpy(), s_new.to_numpy(), rtol=1e-6, atol=1e-8), f"PCA congelado difiere en {n}"
    print("verif 1 OK: transformacion congelada == fit_block en los 3 bloques")

    # 2 un dia nuevo transformado solo da lo mismo que dentro del lote (no depende de estadisticas del lote)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(df), 5, replace=False):
        one, batch = df.iloc[[i]], df.iloc[max(0, i - 40): i + 1]
        for n in BLOCKS:
            a = transform_block(one, art["blocks"][n]).to_numpy()
            b = transform_block(batch, art["blocks"][n]).iloc[[-1]].to_numpy()
            assert np.allclose(a, b, rtol=1e-6, atol=1e-8), f"depende del lote en {n}"
    print("verif 2 OK: dia nuevo aislado == mismo dia dentro de un lote")

    # 3 modelos recargados predicen lo mismo que la prediccion armada con los objetos recargados
    last = df[df["date"] == df["date"].max()]
    for (tgt, h), m in models.items():
        p = predict_dia(last, tgt, h, art, models)
        X = build_X(last, art, art["feats"], m["rep"], m["level"])[m["columns"]]
        raw = m["model"].predict_proba(X)[:, 1] if m["kind"] == "clf" else m["model"].predict(X)
        got = p.iloc[:, -2].to_numpy()
        want = raw if m["kind"] == "clf" else raw + clim_value(last, art["target_clim"][tgt], pd.Index(art["munis"]), h)
        assert np.allclose(got, want), f"prediccion distinta en {tgt} h={h}"
    print(f"verif 3 OK: {len(models)} modelos recargados predicen igual para {df['date'].max().date()}")
    # 4 los objetivos sin modelo caen a climatologia
    for (tgt, h), c in CONFIG.items():
        if c["rep"] is None:
            assert (predict_dia(last, tgt, h, art, models)["fuente"] == "climatologia").all()
    print("verif 4 OK: horizontes sin modelo util usan climatologia")


def main() -> None:   
    #   python congelar_pca.py                       # congela, guarda y verifica
    #   python congelar_pca.py --n-estimators 30     # prueba rapida
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None, help="carpeta raiz de modelos (default: <repo>/models)")
    ap.add_argument("--n-estimators", type=int, default=None, help="solo para pruebas rapidas")
    a = ap.parse_args()
    if a.n_estimators:
        for p in PARAMS.values():
            p["n_estimators"] = a.n_estimators

    path = Path(a.data) if a.data else T.default_master_path()
    feats = master_features(path)
    check_cols(feats)
    df, _ = T.load_data(path)
    df["year"] = df["date"].dt.year

    version = f"v{date.today():%Y%m%d}_{hashlib.sha256(path.read_bytes()).hexdigest()[:8]}"
    root = Path(a.out) if a.out else Path(__file__).resolve().parents[2] / "models"
    out_dir = root / version
    art = freeze_all(df, feats)
    print(f"k por nivel: " + str({lv: {n: k_for(b, lv) for n, b in art['blocks'].items()} for lv in (0.90, 0.95)}))
    models = train_models(df, art, feats)
    save_all(art, models, out_dir, path, version)
    print(f"\nguardado en {out_dir}")
    verify(df, feats, out_dir)


if __name__ == "__main__":
    main()