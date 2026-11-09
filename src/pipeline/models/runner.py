import argparse
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model_config as cfg
import plots
from dataset import build_dataset
from evaluation import compute_metrics, model_scores

IMPORTANCE_SAMPLE = 1000


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str                # "classification" or "regression"
    model_type: str
    estimator: Callable      # returns an unfitted scikit-learn estimator
    params: dict
    complexity: str          # Low/Medium/High
    interpretability: str    # High/Medium/Low
    deployment: str          # deployment difficulty: Low/Medium/High
    data_needs: str

    @property
    def slug(self) -> str:
        prefix = "clf" if self.task == "classification" else "reg"
        name = re.sub(r"\W+", "_", self.name.lower()).strip("_")
        return f"{prefix}_{name}"


def _pipeline(spec: ModelSpec):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), spec.estimator())


def _timed(func, *args):
    start = perf_counter()
    result = func(*args)
    return result, perf_counter() - start


def _importance(model, X, y, task) -> pd.DataFrame:
    sample = X.sample(min(IMPORTANCE_SAMPLE, len(X)), random_state=cfg.RANDOM_STATE)
    scoring = "roc_auc" if task == "classification" else "neg_mean_absolute_error"
    result = permutation_importance(
        model, sample, y.loc[sample.index], scoring=scoring, n_repeats=3, random_state=cfg.RANDOM_STATE
    )
    table = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    return table.sort_values("importance_mean", ascending=False)


def _run_fold(spec, data, year):
    train, test = (data.year < year).to_numpy(), (data.year == year).to_numpy()
    if not train.any() or not test.any():
        return None
    X_test, y_test = data.X[test], data.y[test]

    model = _pipeline(spec)
    _, fit_s = _timed(model.fit, data.X[train], data.y[train])
    pred, predict_s = _timed(model.predict, X_test)
    score = model_scores(model, X_test) if spec.task == "classification" else pred

    predictions = pd.DataFrame({"fold": year, "y_true": y_test.to_numpy(), "y_pred": pred, "y_score": score})
    row = {
        "fold": year, "n_train": int(train.sum()), "n_test": int(test.sum()),
        "fit_seconds": fit_s, "predict_seconds": predict_s,
        **compute_metrics(spec.task, y_test, pred, score),
    }
    if spec.task == "regression":
        predictions["persistence"] = data.persistence[test].to_numpy()
        row["persistence_mae"] = (predictions["persistence"] - predictions["y_true"]).abs().mean()
    return row, predictions, model, X_test, y_test


def _save_plots(spec, predictions, out):
    title, y, pred = spec.name, predictions["y_true"], predictions["y_pred"]
    if spec.task == "classification":
        plots.plot_confusion_matrix(y, pred, title, out / f"{spec.slug}_confusion_matrix.png")
        plots.plot_roc(y, predictions["y_score"], title, out / f"{spec.slug}_roc.png")
    else:
        plots.plot_pred_vs_actual(y, pred, title, out / f"{spec.slug}_pred_vs_actual.png")


def run_model(spec: ModelSpec, table: str = "master") -> None:
    data = build_dataset(spec.task, table)
    out = cfg.output_dir(table)

    rows, frames, last = [], [], None
    for year in cfg.FOLD_YEARS:
        result = _run_fold(spec, data, year)
        if result is None:
            continue
        row, predictions, *last = result
        rows.append(row)
        frames.append(predictions)
        print(f"[{spec.name}] fold {year}: fit {row['fit_seconds']:.1f}s, n_test={row['n_test']}")

    folds, predictions = pd.DataFrame(rows), pd.concat(frames, ignore_index=True)
    pooled = compute_metrics(spec.task, predictions["y_true"], predictions["y_pred"], predictions["y_score"])
    if spec.task == "classification":
        baseline = {"positive_rate": float(predictions["y_true"].mean())}
    else:
        baseline = {"persistence_mae": float((predictions["persistence"] - predictions["y_true"]).abs().mean())}

    model, X_last, y_last = last
    importance = _importance(model, X_last, y_last, spec.task)

    summary = {
        "name": spec.name, "task": spec.task, "slug": spec.slug, "table": table,
        "model_type": spec.model_type, "params": spec.params,
        "complexity": spec.complexity, "interpretability": spec.interpretability,
        "deployment": spec.deployment, "data_needs": spec.data_needs,
        "n_features": data.X.shape[1], "n_rows": len(data.X),
        "fit_seconds": float(folds["fit_seconds"].sum()),
        "predict_seconds": float(folds["predict_seconds"].sum()),
        "pooled": {k: float(v) for k, v in pooled.items()},
        "baseline": baseline,
    }
    folds.to_csv(out / f"{spec.slug}_folds.csv", index=False)
    predictions.to_parquet(out / f"{spec.slug}_predictions.parquet", index=False)
    importance.to_csv(out / f"{spec.slug}_importance.csv", index=False)
    (out / f"{spec.slug}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _save_plots(spec, predictions, out)
    print(f"[{spec.name}] done -> {out}")


def main(spec: ModelSpec) -> None:
    """Entry point for each model script."""
    parser = argparse.ArgumentParser(description=f"Train and evaluate: {spec.name}")
    parser.add_argument("--table", choices=sorted(cfg.TABLES), default="master")
    run_model(spec, parser.parse_args().table)
