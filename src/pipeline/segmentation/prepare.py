import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seg_config as cfg
from common import write_json


def monthly(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp())
    g = df.groupby(["municipio", "month"], observed=True)
    agg = g[cfg.VARS].mean()
    agg[cfg.SUM_VARS] = g[cfg.SUM_VARS].sum(min_count=1)
    agg["n_days"] = g.size()
    return agg.reset_index()


def main():
    cfg.OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(cfg.MASTER)
    missing = [c for c in cfg.VARS if c not in df.columns]
    if missing:
        numeric = [c for c in df.select_dtypes("number").columns if "aod" not in c.lower()]
        raise SystemExit(f"Missing columns {missing} in {cfg.MASTER}. Available non-AOD numeric columns: {numeric}")
    df["date"] = pd.to_datetime(df["date"])

    full = monthly(df)
    month = full[full["n_days"] >= cfg.MIN_DAYS].dropna(subset=cfg.VARS).reset_index(drop=True)
    month.to_parquet(cfg.OUT / "monthly_raw.parquet", index=False)

    z = StandardScaler().fit_transform(month[cfg.VARS])
    pca = PCA(n_components=cfg.PCA_VAR, svd_solver="full", random_state=cfg.SEED).fit(z)
    names = [f"PC{i + 1}" for i in range(pca.n_components_)]
    scores = pd.DataFrame(pca.transform(z), columns=names)
    scores.insert(0, "month", month["month"])
    scores.insert(0, "municipio", month["municipio"])
    scores.to_parquet(cfg.OUT / "pca_scores.parquet", index=False)
    pd.DataFrame(pca.components_.T, index=cfg.VARS, columns=names).round(3).to_csv(cfg.OUT / "pca_loadings.csv")

    explained = pca.explained_variance_ratio_
    write_json("prep_meta.json", {
        "n_rows_master": len(df), "n_cols_master": df.shape[1],
        "n_obs": len(month), "n_months_dropped": len(full) - len(month), "n_vars": len(cfg.VARS),
        "vars": cfg.VARS, "sum_vars": cfg.SUM_VARS, "min_days": cfg.MIN_DAYS,
        "municipios": month["municipio"].nunique(),
        "period": [str(month["month"].min().date()), str(month["month"].max().date())],
        "pca_var": cfg.PCA_VAR, "n_pc": int(pca.n_components_),
        "explained": explained.tolist(), "cumulative": np.cumsum(explained).tolist(),
        "eigenvalues": pca.explained_variance_.tolist(), "kaiser": int((pca.explained_variance_ > 1).sum()),
    })

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, explained, color="steelblue", label="Explained variance")
    ax.plot(names, np.cumsum(explained), "o-", color="darkorange", label="Cumulative")
    ax.axhline(cfg.PCA_VAR, ls="--", color="gray")
    ax.set_ylabel("Variance ratio")
    ax.set_title("PCA scree plot (municipio-month)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.OUT / "pca_scree.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()