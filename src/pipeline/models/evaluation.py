import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, mean_squared_error,precision_score, r2_score, recall_score, roc_auc_score,)


def model_scores(model, X):
    # sacar los scores de un modelo entrenado, para classification y regression
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(task: str, y_true, y_pred, y_score=None) -> dict:
    if task == "classification":
        has_both_classes = len(np.unique(y_true)) > 1
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_score) if has_both_classes else float("nan"),
        }
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
    }
