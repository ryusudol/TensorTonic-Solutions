import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    W, b = np.zeros(n_features, dtype=float), 0.0

    for _ in range(steps):
        p = _sigmoid(X @ W + b)
        delta = p - y
        W -= lr * (X.T @ delta) / n_samples
        b -= lr * float(np.mean(delta))

    return (W, b)