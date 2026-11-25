import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import params as P
from model import MultiTaskLSTM


def loss_fn(out, y, targets):
    total = 0.0
    for i, t in enumerate(targets):
        spec = P.TARGETS[t]
        fn = F.binary_cross_entropy_with_logits if spec["kind"] == "clf" else F.mse_loss
        total = total + spec["weight"] * fn(out[:, i], y[:, i])
    return total


def fit(fold, n_features, targets, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskLSTM(n_features, targets).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=P.LR)
    loader = DataLoader(TensorDataset(torch.from_numpy(fold["Xtr"]), torch.from_numpy(fold["Ytr"])),
                        batch_size=P.BATCH, shuffle=True)
    Xv, Yv = torch.from_numpy(fold["Xva"]).to(dev), torch.from_numpy(fold["Yva"]).to(dev)

    hist, best, best_state, bad, t0 = {"train_loss": [], "val_loss": []}, float("inf"), None, 0, time.time()
    for _ in range(P.EPOCHS):
        model.train()
        run, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb, targets)
            loss.backward()
            opt.step()
            run, n = run + loss.item() * len(xb), n + len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), Yv, targets).item()
        hist["train_loss"].append(run / n)
        hist["val_loss"].append(val)
        if val < best:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= P.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, hist, time.time() - t0


@torch.no_grad()
def predict(model, X, targets):
    model.eval()
    dev = next(model.parameters()).device
    out = np.concatenate([model(torch.from_numpy(X[i:i + 2048]).to(dev)).cpu().numpy()
                          for i in range(0, len(X), 2048)])
    for i, t in enumerate(targets):
        if P.TARGETS[t]["kind"] == "clf":
            out[:, i] = 1 / (1 + np.exp(-out[:, i]))
    return out


def plot_curves(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(histories), figsize=(4 * len(histories), 3.2), squeeze=False)
    for ax, (year, h) in zip(axs[0], histories.items()):
        ax.plot(h["train_loss"], label="train")
        ax.plot(h["val_loss"], label="val")
        ax.set_title(f"test {year}")
        ax.set_xlabel("epoch")
    axs[0][0].set_ylabel("loss")
    axs[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)