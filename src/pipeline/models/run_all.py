import argparse
import importlib

import model_config as cfg
from report import write_report
from runner import run_model

MODEL_MODULES = (
    "naive_bayes", "svm_classifier", "mlp_classifier",
    "linear_regression", "svr", "mlp_regressor",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full model pipeline")
    parser.add_argument("--table", choices=sorted(cfg.TABLES), default="master")
    table = parser.parse_args().table

    for module in MODEL_MODULES:
        run_model(importlib.import_module(module).SPEC, table)
    print(f"Report: {write_report(table)}")


if __name__ == "__main__":
    main()
