import json
import numpy as np
import pandas as pd
import torch
import cfg, common
from cfg import P
from deep_learning import train


def losses(pred, Y, targets):
    """Per-row loss per target: BCE (clf) or squared error on the standardized target (reg)."""
    out = {}
    for i, t in enumerate(targets):
        if P.TARGETS[t]["kind"] == "clf":
            p = np.clip(pred[:, i], 1e-6, 1 - 1e-6)
            out[t] = -(Y[:, i] * np.log(p) + (1 - Y[:, i]) * np.log(1 - p))
        else:
            out[t] = (pred[:, i] - Y[:, i]) ** 2
    return out


def build_units(feats):
    W, units = P.WINDOW, {f: (slice(None), i) for i, f in enumerate(feats)}
    for g, keys in cfg.FEATURE_GROUPS.items():
        idx = [i for i, f in enumerate(feats) if any(k in f.lower() for k in keys)]
        if idx:
            units[f"group:{g}"] = (slice(None), idx)
    for a, b in cfg.LAG_BLOCKS:
        units[f"lag:{a}-{b}"] = (slice(W - b, W - a + 1), slice(None))
    return units


def permutation(model, X, Y, units, rng, targets):
    base = {t: v.mean() for t, v in losses(train.predict(model, X, targets), Y, targets).items()}
    res = {t: {} for t in base}
    for name, (ts, fi) in units.items():
        inc = {t: [] for t in base}
        for _ in range(cfg.PERM_REPEATS):
            Xp = X.copy()
            Xp[:, ts, fi] = X[:, ts, fi][rng.permutation(len(X))]
            for t, v in losses(train.predict(model, Xp, targets), Y, targets).items():
                inc[t].append(v.mean() - base[t])
        for t in base:
            res[t][name] = (float(np.mean(inc[t])), float(np.std(inc[t])))
    return base, res


def grad_x_input(model, X, targets, n=1000):
    dev = next(model.parameters()).device
    x = torch.from_numpy(X[:n]).to(dev).requires_grad_(True)
    model.eval()
    res = {}
    with torch.backends.cudnn.flags(enabled=False):          # cuDNN RNN backward needs train mode otherwise
        out = model(x)
        for i, t in enumerate(targets):
            g, = torch.autograd.grad(out[:, i].sum(), x, retain_graph=True)
            res[t] = (g * x).abs().mean(dim=0).detach().cpu().numpy()      # (window, features)
    return res


def plot(t, top, lags, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    axs[0].barh([k for k, _ in top][::-1], [v[0] for _, v in top][::-1])
    axs[0].set_title(f"{t}: top features (loss increase)")
    axs[1].bar(list(lags), [v[0] for v in lags.values()])
    axs[1].set_title("lag blocks (days)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    gate = json.loads((cfg.OUT_DIR / "gate.json").read_text())
    targets = [t for t, g in gate.items() if g.get("eligible")]
    forced = False
    if not targets and cfg.FORCE_XAI:
        targets, forced = list(P.TARGETS), True
    if not targets:
        (cfg.OUT_DIR / "explain.json").write_text(json.dumps(
            {"skipped": "no target passed the gate; set cfg.FORCE_XAI=True for diagnostics"}))
        return
    df, feats = common.load()
    r = common.run_fold(df, feats, cfg.XAI_YEAR, targets, cfg.XAI_SEED)
    T, fold, model = r["targets"], r["fold"], r["model"]
    X, Y = fold["Xte"], fold["Yte"]
    base, perm = permutation(model, X, Y, build_units(feats), np.random.default_rng(cfg.XAI_SEED), T)
    gxi = grad_x_input(model, X, T)
    row_loss = losses(r["pred"], Y, T)

    out = dict(year=cfg.XAI_YEAR, seed=cfg.XAI_SEED, targets=T, forced=forced)
    for t in T:
        feat = {k: v for k, v in perm[t].items() if k in feats}
        top = sorted(feat.items(), key=lambda kv: -kv[1][0])[:cfg.TOP_K]
        lags = {k: v for k, v in perm[t].items() if k.startswith("lag:")}
        groups = {k: v for k, v in perm[t].items() if k.startswith("group:")}
        pi = pd.Series({k: v[0] for k, v in feat.items()})
        gi = pd.Series(gxi[t].mean(axis=0), index=feats)
        rows = pd.DataFrame({"month": pd.DatetimeIndex(fold["dates_te"]).month, "municipio": fold["mun_te"], "loss": row_loss[t]})
        out[t] = dict(baseline_loss=float(base[t]),
                      top_features=[dict(feature=k, increase=v[0], std=v[1]) for k, v in top],
                      groups={k: dict(increase=v[0], std=v[1]) for k, v in groups.items()},
                      lag_blocks={k: dict(increase=v[0], std=v[1]) for k, v in lags.items()},
                      spearman_vs_gradxinput=float(pi.corr(gi.reindex(pi.index), method="spearman")),
                      loss_by_month=rows.groupby("month")["loss"].mean().to_dict(),
                      loss_by_municipio=rows.groupby("municipio", observed=True)["loss"].mean().to_dict())
        plot(t, top, lags, cfg.FIG_DIR / f"xai_{t}.png")
    (cfg.OUT_DIR / "explain.json").write_text(json.dumps(out, indent=2, default=float))