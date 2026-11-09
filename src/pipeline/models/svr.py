from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from runner import ModelSpec, main

PARAMS = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}  # epsilon in standardized-target units

SPEC = ModelSpec(
    name="Support Vector Regression",
    task="regression",
    model_type="Kernel margin regressor",
    estimator=lambda: TransformedTargetRegressor(regressor=SVR(**PARAMS), transformer=StandardScaler()),
    params=PARAMS,
    complexity="Medium",
    interpretability="Low",
    deployment="Medium",
    data_needs="Medium: training cost grows roughly quadratically with rows; needs scaled features and target.",
)

if __name__ == "__main__":
    main(SPEC)
