from sklearn.svm import SVC
from runner import ModelSpec, main

PARAMS = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced"}

SPEC = ModelSpec(
    name="SVM",
    task="classification",
    model_type="Kernel margin classifier",
    estimator=lambda: SVC(**PARAMS),
    params=PARAMS,
    complexity="Medium",
    interpretability="Low",
    deployment="Medium",
    data_needs="Medium: training cost grows roughly quadratically with rows; needs scaled features.",
)

if __name__ == "__main__":
    main(SPEC)
