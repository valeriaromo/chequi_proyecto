"""Linear Regression: next-day minimum temperature (T2M_MIN)."""
from sklearn.linear_model import LinearRegression

from runner import ModelSpec, main

PARAMS = {"fit_intercept": True}

SPEC = ModelSpec(
    name="Linear Regression",
    task="regression",
    model_type="Linear (OLS)",
    estimator=lambda: LinearRegression(**PARAMS),
    params=PARAMS,
    complexity="Low",
    interpretability="High",
    deployment="Low",
    data_needs="Low: closed-form fit; unstable coefficients with highly collinear lags.",
)

if __name__ == "__main__":
    main(SPEC)
