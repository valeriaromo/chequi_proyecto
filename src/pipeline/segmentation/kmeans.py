import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seg_config as cfg
from common import pc_cols, write_json


def elbow_k(ks, inertia):
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(ks[np.argmax(np.abs(x + y - 1))])


def note(k, row, n, elbow, best, final):
    tags = []
    if k == final:
        tags.append("**SELECTED**")
    if k == elbow:
        tags.append("elbow")
    if k == best:
        tags.append(f"best silhouette among k ≥ {cfg.K_FINAL_MIN}")
    if k < cfg.K_FINAL_MIN:
        tags.append("too coarse: likely a single seasonal split")
    if row["min_size"] / n < 0.03:
        tags.append("contains a cluster < 3% of observations")
    return "; ".join(tags) or "-"


def main(k_override=None):
    scores = pd.read_parquet(cfg.OUT / "pca_scores.parquet")
    x = scores[pc_cols(scores)].to_numpy()
    rows = []
    for k in cfg.K_RANGE:
        km = KMeans(k, n_init=10, random_state=cfg.SEED).fit(x)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(x, km.labels_),
                     "min_size": int(np.bincount(km.labels_).min())})
    res = pd.DataFrame(rows)
    ks, inertia = res["k"].to_numpy(), res["inertia"].to_numpy()
    elbow = elbow_k(ks, inertia)
    eligible = res[res["k"] >= cfg.K_FINAL_MIN]
    best = int(eligible.loc[eligible["silhouette"].idxmax(), "k"])
    final = k_override or best

    res["interpretation"] = [note(r.k, r._asdict(), len(x), elbow, best, final) for r in res.itertuples()]
    res.to_csv(cfg.OUT / "kmeans_configs.csv", index=False)

    km = KMeans(final, n_init=10, random_state=cfg.SEED).fit(x)
    order = pd.Series(km.labels_).value_counts().index  # relabel so cluster 0 is the largest
    mapping = {old: new for new, old in enumerate(order)}
    labels = scores[["municipio", "month"]].assign(cluster=pd.Series(km.labels_).map(mapping).to_numpy())
    labels.to_parquet(cfg.OUT / "labels.parquet", index=False)
    sizes = labels["cluster"].value_counts().sort_index()
    pd.DataFrame({"cluster": sizes.index, "n": sizes.values, "share": sizes.values / len(labels)}).to_csv(
        cfg.OUT / "cluster_assignment.csv", index=False)
    write_json("kmeans_meta.json", {"k_final": final, "k_elbow": elbow, "k_best_silhouette": best,"silhouette_final": float(res.loc[res["k"] == final, "silhouette"].iloc[0])})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, inertia, "o-")
    axes[0].axvline(elbow, ls="--", color="gray", label=f"elbow k={elbow}")
    axes[0].set(xlabel="K", ylabel="Inertia", title="Elbow method")
    axes[0].legend()
    axes[1].plot(ks, res["silhouette"], "o-", color="darkorange")
    axes[1].axvline(final, ls="--", color="gray", label=f"selected k={final}")
    axes[1].set(xlabel="K", ylabel="Silhouette", title="Silhouette score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(cfg.OUT / "kmeans_selection.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()