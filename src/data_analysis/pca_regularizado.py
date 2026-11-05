"""Misma comparacion de pca_walkforward.py pero con XGBoost mas regularizado, solo h>=3.
Guarda en otra carpeta para no pisar los resultados anteriores."""
import sys
import test_pca as T
import pca_walkforward as W

T.XGB = {**T.XGB, "n_estimators": 150, "max_depth": 3, "learning_rate": 0.03,
         "min_child_weight": 50, "reg_lambda": 10.0}          # mismos hiperparametros para todas las variantes
sys.argv = [sys.argv[0], "--horizons", "3", "7", "--targets", "T2M_MAX", "T2M_MIN",
            "--levels", "0.95", "--out", str(T.OUT_DIR / "regularizado"), *sys.argv[1:]]
W.main()