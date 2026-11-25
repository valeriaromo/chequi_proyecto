import json
import time
import torch
import params as P
import data, evaluate, report, train


def main():
    T = list(P.ACTIVE_TARGETS)
    P.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = data.load_master()
    feats = data.feature_cols(df)
    df = data.add_targets(data.ffill_features(df, feats), T)

    res = {"n_features": len(feats), "folds": {}, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    histories = {}
    for year in P.TEST_YEARS:
        fold = data.build_fold(df, feats, year, T)
        model, hist, secs = train.fit(fold, len(feats), T, P.SEED)
        t0 = time.time()
        pred = train.predict(model, fold["Xte"], T)
        res["infer_ms"] = (time.time() - t0) * 1000 / max(len(fold["Xte"]), 1) * 1000
        res["params"] = sum(p.numel() for p in model.parameters())
        res["folds"][str(year)] = dict(epochs=len(hist["val_loss"]), seconds=secs,
                                       train_loss=hist["train_loss"][-1], val_loss=min(hist["val_loss"]),
                                       n_test=int(len(fold["Xte"])),
                                       metrics=evaluate.evaluate_fold(fold, pred, T))
        histories[str(year)] = hist
        print(f"{year}: {res['folds'][str(year)]['metrics']}")

    json.dump(res, open(P.OUT_DIR / "results.json", "w"), indent=2, default=float)
    json.dump(histories, open(P.OUT_DIR / "history.json", "w"), indent=2)
    train.plot_curves(histories, P.FIG_DIR / "learning_curves.png")
    report.build(res)


if __name__ == "__main__":
    main()