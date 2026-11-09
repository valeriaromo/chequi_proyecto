from sklearn.naive_bayes import GaussianNB
from runner import ModelSpec, main
PARAMS = {"var_smoothing": 1e-9}

SPEC = ModelSpec(
    name="Naive Bayes",
    task="classification",
    model_type="Probabilistic (Gaussian)",
    estimator=lambda: GaussianNB(**PARAMS),
    params=PARAMS,
    complexity="Low",
    interpretability="High",
    deployment="Low",
    data_needs="Low: only per-class means and variances; assumes feature independence, which lags and rolling means violate.",
)

if __name__ == "__main__":
    main(SPEC)
