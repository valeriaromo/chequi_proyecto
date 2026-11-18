import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seg_config as cfg
from common import load, pc_cols, read_json
CMAP = plt.get_cmap("tab10")

def save(fig, name):
    fig.tight_layout()
    fig.savefig(cfg.OUT / name, dpi=150)
    plt.close(fig)

def heatmap(m: pd.DataFrame, title, name, fmt="{:.2f}", cmap="viridis", center=False, size=(8, 5)):
    fig, ax = plt.subplots(figsize=size)
    lim = np.abs(m.to_numpy()).max() if center else None
    im = ax.imshow(m.to_numpy(), cmap=cmap, aspect="auto", vmin=-lim if center else None, vmax=lim)
    ax.set_xticks(range(m.shape[1]), [str(c) for c in m.columns], rotation=45, ha="right")
    ax.set_yticks(range(m.shape[0]), [str(i) for i in m.index])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, fmt.format(m.iat[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    save(fig, name)

def pca_scatter(d, ev):
    fig, ax = plt.subplots(figsize=(8, 6))
    for c, g in d.groupby("cluster"):
        ax.scatter(g["PC1"], g["PC2"], s=14, alpha=0.6, color=CMAP(c), label=f"C{c} (n={len(g)})")
    cen = d.groupby("cluster")[["PC1", "PC2"]].mean()
    ax.scatter(cen["PC1"], cen["PC2"], marker="X", s=220, c="k", edgecolor="w", label="Centroids")
    ax.set(xlabel=f"PC1 ({ev[0]:.1%})", ylabel=f"PC2 ({ev[1]:.1%})", title="Clusters in PCA space")
    ax.legend()
    save(fig, "pca_scatter.png")

def pair_plot(d):
    pcs = pc_cols(d)[:3]
    fig, axes = plt.subplots(len(pcs), len(pcs), figsize=(9, 9))
    for i, a in enumerate(pcs):
        for j, b in enumerate(pcs):
            ax = axes[i, j]
            for c, g in d.groupby("cluster"):
                if i == j:
                    ax.hist(g[a], bins=20, alpha=0.5, color=CMAP(c))
                else:
                    ax.scatter(g[b], g[a], s=6, alpha=0.5, color=CMAP(c))
            if i == len(pcs) - 1:
                ax.set_xlabel(b)
            if j == 0:
                ax.set_ylabel(a)
    fig.suptitle("Pair plot of the first principal components by cluster")
    save(fig, "pair_plot.png")


def main():
    d = load()
    ev = read_json("prep_meta.json")["explained"]
    d["cal_month"] = d["month"].dt.month
    k = d["cluster"].nunique()

    pca_scatter(d, ev)
    pair_plot(d)

    z = pd.read_csv(cfg.OUT / "profile_z.csv", index_col=0)
    z.index = [f"C{c}" for c in z.index]
    heatmap(z, "Cluster centroids (z-score of each variable)", "centroid_heatmap.png", cmap="RdBu_r", center=True, size=(9, 1.2 + 0.7 * k))

    sizes = d["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"C{c}" for c in sizes.index], sizes.values, color=[CMAP(c) for c in sizes.index])
    for i, v in enumerate(sizes.values):
        ax.text(i, v, f"{v} ({v / len(d):.0%})", ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Observations", title="Cluster sizes")
    save(fig, "cluster_sizes.png")

    by_month = pd.crosstab(d["cluster"], d["cal_month"], normalize="index").reindex(columns=range(1, 13), fill_value=0)
    by_month.index = [f"C{c}" for c in by_month.index]
    heatmap(by_month, "Calendar-month distribution inside each cluster", "cluster_by_month.png", fmt="{:.0%}", size=(10, 1.2 + 0.7 * k))

    by_muni = pd.crosstab(d["municipio"], d["cluster"], normalize="index")
    by_muni.columns = [f"C{c}" for c in by_muni.columns]
    heatmap(by_muni, "Cluster mix of each municipality", "cluster_by_municipio.png", fmt="{:.0%}", size=(8, 1 + 0.5 * len(by_muni)))


if __name__ == "__main__":
    main()