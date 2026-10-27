from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

AOD = "aod_055_mean"
NASA = "NASA POWER daily series at the municipal centroid"
FEATURES = {
    "T2M": ("Mean 2 m air temperature (°C)", NASA, "mean temperature"),
    "T2M_MAX": ("Daily maximum 2 m air temperature (°C)", NASA, "daytime maximum temperature"),
    "T2M_MIN": ("Daily minimum 2 m air temperature (°C); pilot target", NASA, "night minimum temperature"),
    "WS2M": ("2 m wind speed (m/s)", NASA, "wind"),
    "RH2M": ("2 m relative humidity (%)", NASA, "humidity"),
    "ALLSKY_SFC_SW_DWN": ("All-sky incoming shortwave radiation", NASA, "solar radiation"),
    AOD: ("Aerosol optical depth at 0.55 µm", "MAIAC via Google Earth Engine, zonal mean over the municipal polygon", "aerosol load"),
    "PRECTOTCORR": ("Bias-corrected daily precipitation (mm)", NASA, "rainfall"),
}
PCA_VARS = list(FEATURES)
N_PC = 3             # components analizados a detalle
VARIANCE_TARGET = 0.80
LOADING_MIN = 0.40   # de que variable es dominante un componente
UNUSUAL_P = 0.999    # chic uadrada para 
SEED = 42


@dataclass
class PCAResult:
    eigenvalues: np.ndarray
    loadings: pd.DataFrame   # variable x PC: correlation entre variable y componente
    scores: pd.DataFrame     # observation x PC

    @property
    def ratio(self) -> np.ndarray:
        return self.eigenvalues / self.eigenvalues.sum()

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.ratio)


def standardise(X: pd.DataFrame) -> pd.DataFrame:
    # zscore con la estadística observada y handle de los NaN
    return ((X - X.mean()) / X.std(ddof=0)).fillna(0.0)


def fit_pca(Z: pd.DataFrame) -> PCAResult:
    # PCA por SVD y datos estandarizados
    values = Z.to_numpy(float)
    _, S, Vt = np.linalg.svd(values, full_matrices=False)
    Vt = Vt * np.sign(Vt[np.arange(len(Vt)), np.abs(Vt).argmax(axis=1)])[:, None]  # signos de rebroductibilidad
    eig = S ** 2 / (len(Z) - 1)
    names = [f"PC{i + 1}" for i in range(len(S))]
    return PCAResult(eig, pd.DataFrame(Vt.T * np.sqrt(eig), index=Z.columns, columns=names),pd.DataFrame(values @ Vt.T, index=Z.index, columns=names))


def components_needed(res: PCAResult, target: float = VARIANCE_TARGET) -> int:
    return int(np.argmax(res.cumulative >= target)) + 1


def loading_similarity(full: PCAResult, reduced: PCAResult, n: int = N_PC) -> np.ndarray:
    # coseno entre los primeros n vectores de los PCA 
    a = full.loadings.loc[reduced.loadings.index].iloc[:, :n].to_numpy()
    b = reduced.loadings.iloc[:, :n].to_numpy()
    return np.abs((a * b).sum(0)) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))


def interpret(load: pd.Series) -> tuple[str, str]:
    # variables domintantes y lectura de cada componente
    dom = load[load.abs() >= LOADING_MIN].sort_values(key=np.abs, ascending=False)
    if dom.empty:
        dom = load.loc[load.abs().nlargest(2).index]
    concepts = lambda mask: sorted({FEATURES[v][2] for v in dom[mask].index})
    pos, neg = concepts(dom > 0), concepts(dom < 0)
    reading = (f"Contrast between {', '.join(pos)} (+) and {', '.join(neg)} (-)" if pos and neg
               else f"Common movement of {', '.join(pos or neg)}")
    return ", ".join(f"{v} ({w:+.2f})" for v, w in dom.items()), reading


def eta_squared(score: pd.Series, groups: pd.Series) -> float:
    g = score.groupby(groups).agg(["mean", "size"])
    between = (g["size"] * (g["mean"] - score.mean()) ** 2).sum()
    return float(between / ((score - score.mean()) ** 2).sum())


def structure_table(scores: pd.DataFrame, meta: pd.DataFrame, n: int = N_PC) -> pd.DataFrame:
    # porcentaje de cada varianza de los PC explicada por municipalidad, mes, año y si jala el foKin AOD
    groupings = {"Municipality": meta.municipio, "Month": meta.month, "Year": meta.yr, "AOD observed": meta.aod_obs}
    return pd.DataFrame({name: {pc: 100 * eta_squared(scores[pc], key) for pc in scores.columns[:n]}
                         for name, key in groupings.items()}).T


def lag1_autocorr(scores: pd.DataFrame, municipio: pd.Series, n: int = N_PC) -> pd.Series:
    # correlacion de las medias por día
    return pd.DataFrame({m: g.iloc[:, :n].apply(lambda s: s.autocorr(1))
                         for m, g in scores.groupby(municipio)}).T.mean()


def shape_stats(scores: pd.DataFrame, n: int = N_PC) -> pd.DataFrame:
    s = scores.iloc[:, :n]
    return pd.DataFrame({"Skewness": s.apply(stats.skew), "Excess_kurtosis": s.apply(stats.kurtosis)})


def unusual_days(res: PCAResult, n: int = N_PC, p: float = UNUSUAL_P) -> tuple[pd.Series, float]:
    # se usa distancia de Mahalanobis al cuadrado para los primeros PC y chi al cuadrado para los inusuales
    d2 = (res.scores.iloc[:, :n] ** 2 / res.eigenvalues[:n]).sum(axis=1)
    return d2, float(stats.chi2.ppf(p, n))


def _save(fig, path: Path) -> str:
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_variance(res: PCAResult, path: Path) -> str:
    x = np.arange(1, len(res.ratio) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    ax.bar(x, res.ratio * 100, color="#3182bd", label="Per component")
    ax.plot(x, res.cumulative * 100, color="#E45756", marker="o", label="Cumulative")
    ax.axhline(VARIANCE_TARGET * 100, ls="--", color="grey", lw=.8)
    ax.set_xticks(x, [f"PC{i}" for i in x])
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)",
           title="Explained variance of the daily anomaly PCA (dashed: 80% target)")
    ax.legend(loc="center right")
    return _save(fig, path)


def fig_loadings(res: PCAResult, path: Path, n: int = N_PC) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), layout="constrained")
    res.loadings.iloc[:, :n].plot.bar(ax=ax, width=.8)
    ax.axhline(0, color="k", lw=.6)
    for s in (-LOADING_MIN, LOADING_MIN):
        ax.axhline(s, ls="--", color="grey", lw=.7)
    ax.set_xticklabels(res.loadings.index, rotation=35, ha="right")
    ax.set(xlabel="Original variable", ylabel="Loading (correlation with component)",
           title="Which variables drive each component? (dashed: |loading| = 0.40)")
    return _save(fig, path)


def fig_projections(res: PCAResult, flag: pd.Series, path: Path) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
        x, y = res.scores.iloc[:, i], res.scores.iloc[:, j]
        hb = ax.hexbin(x, y, gridsize=45, bins="log", mincnt=1, cmap="viridis")
        ax.scatter(x[flag], y[flag], s=8, color="#E45756", label="Unusual days")
        ax.set(xlabel=f"PC{i + 1} ({res.ratio[i] * 100:.1f}% variance)",
               ylabel=f"PC{j + 1} ({res.ratio[j] * 100:.1f}% variance)", title=f"PC{i + 1} vs PC{j + 1}")
        ax.legend(fontsize=7)
    fig.colorbar(hb, ax=axes.tolist(), label="Days (log count)")
    fig.suptitle("Two-component projections of daily anomalies", fontweight="bold")
    return _save(fig, path)


def fig_by_month(res: PCAResult, months: pd.Series, path: Path) -> str:
    idx = np.random.default_rng(SEED).choice(len(months), size=min(6000, len(months)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    sc = ax.scatter(res.scores.PC1.to_numpy()[idx], res.scores.PC2.to_numpy()[idx],
                    c=months.to_numpy()[idx], cmap="twilight", s=6)
    fig.colorbar(sc, ax=ax, label="Month")
    ax.set(xlabel=f"PC1 ({res.ratio[0] * 100:.1f}% variance)", ylabel=f"PC2 ({res.ratio[1] * 100:.1f}% variance)",
           title="PC1 vs PC2 coloured by month: is seasonality left?")
    return _save(fig, path)