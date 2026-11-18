import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_samples)
import seg_config as cfg
from common import load, pc_cols, read_json, write_json


def eta2(values: pd.Series, labels: pd.Series) -> float:
    total = ((values - values.mean()) ** 2).sum()
    between = sum(len(g) * (g.mean() - values.mean()) ** 2 for _, g in values.groupby(labels))
    return float(between / total)


def main():
    d = load()
    k = read_json("kmeans_meta.json")["k_final"]
    d["cal_month"] = d["month"].dt.month
    z = (d[cfg.VARS] - d[cfg.VARS].mean()) / d[cfg.VARS].std()

    profile = d.groupby("cluster")[cfg.VARS].mean()
    zprofile = z.groupby(d["cluster"]).mean()
    eta = pd.Series({v: eta2(d[v], d["cluster"]) for v in cfg.VARS}).sort_values(ascending=False)
    profile.to_csv(cfg.OUT / "profile.csv")
    zprofile.to_csv(cfg.OUT / "profile_z.csv")
    eta.rename("eta2").to_csv(cfg.OUT / "eta2.csv")

    chars = {}
    for c, g in d.groupby("cluster"):
        zc = zprofile.loc[c]
        top = zc.reindex(zc.abs().sort_values(ascending=False).index)[:3]
        muni = g["municipio"].value_counts(normalize=True)
        chars[int(c)] = {
            "n": len(g), "share": len(g) / len(d),
            "top_vars": [(v, float(s)) for v, s in top.items()],
            "top_months": {int(m): float(s) for m, s in g["cal_month"].value_counts(normalize=True).head(3).items()},
            "n_municipios": int(g["municipio"].nunique()), "max_muni_share": float(muni.iloc[0]),
        }
    write_json("characterization.json", chars)

    x, y = d[pc_cols(d)].to_numpy(), d["cluster"].to_numpy()
    sil = silhouette_samples(x, y)
    per_cluster = pd.Series(sil).groupby(y).mean()
    rng = np.random.default_rng(cfg.SEED)
    aris = []
    for i in range(cfg.BOOT_RUNS):  # stability: fit on 80% of the data, predict all, compare with the final labels
        idx = rng.choice(len(x), int(0.8 * len(x)), replace=False)
        km = KMeans(k, n_init=10, random_state=i).fit(x[idx])
        aris.append(adjusted_rand_score(y, km.predict(x)))

    write_json("validation.json", {
        "silhouette": float(sil.mean()),
        "silhouette_per_cluster": {int(c): float(v) for c, v in per_cluster.items()},
        "neg_silhouette_share": float((sil < 0).mean()),
        "davies_bouldin": float(davies_bouldin_score(x, y)),
        "calinski_harabasz": float(calinski_harabasz_score(x, y)),
        "ari_mean": float(np.mean(aris)), "ari_std": float(np.std(aris)),
        "nmi_month": float(normalized_mutual_info_score(y, d["cal_month"])),
        "nmi_municipio": float(normalized_mutual_info_score(y, d["municipio"])),
        "avg_clusters_per_municipio": float(d.groupby("municipio", observed=True)["cluster"].nunique().mean()),
        "pc_eta2": {pc: eta2(d[pc], d["cluster"]) for pc in pc_cols(d)},
        "min_share": float(min(c["share"] for c in chars.values())),
    })


if __name__ == "__main__":
    main()