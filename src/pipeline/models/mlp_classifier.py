from sklearn.neural_network import MLPClassifier
from model_config import RANDOM_STATE
from runner import ModelSpec, main

PARAMS = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "alpha": 1e-4,
    "early_stopping": True,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}

SPEC = ModelSpec(
    name="Neural Network",
    task="classification",
    model_type="Multilayer perceptron",
    estimator=lambda: MLPClassifier(**PARAMS),
    params=PARAMS,
    complexity="High",
    interpretability="Low",
    deployment="Medium",
    data_needs="High: benefits from many rows; sensitive to scaling and random initialization.",
)

if __name__ == "__main__":
    main(SPEC)
