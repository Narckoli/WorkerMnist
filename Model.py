# model.py
import numpy as np
from typing import Dict, Tuple

# ======================
# Activaciones
# ======================
def relu(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0, Z)

def relu_derivative(Z: np.ndarray) -> np.ndarray:
    return (Z > 0).astype(float)

def softmax(Z: np.ndarray) -> np.ndarray:
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def one_hot(y: np.ndarray) -> np.ndarray:
    oh = np.zeros((y.size, 10))
    oh[np.arange(y.size), y] = 1
    return oh

# ======================
# Forward
# ======================
def forward(X: np.ndarray, weights: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple]:
    """Forward pass de la red."""
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]

    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    cache = (X, Z1, A1)
    return A2, cache

# ======================
# Loss
# ======================
def compute_loss(A2: np.ndarray, y: np.ndarray) -> float:
    """Calcula la loss (cross-entropy)."""
    m = y.shape[0]
    log_likelihood = -np.log(A2[np.arange(m), y] + 1e-9)
    return np.sum(log_likelihood) / m

# ======================
# Backward
# ======================
def backward(A2: np.ndarray, y: np.ndarray, cache: Tuple, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Backward pass - calcula gradientes."""
    X, Z1, A1 = cache
    W2 = weights["W2"]
    m = y.shape[0]
    Y = one_hot(y)

    # Capa de salida
    dZ2 = A2 - Y
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0)

    # Capa oculta
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }