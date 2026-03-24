# model.py (worker)
import numpy as np
from typing import Dict, Tuple

def init_local_weights(input_size: int, hidden_size: int, output_size: int) -> Dict[str, np.ndarray]:
    """Inicializa la estructura de pesos para gradientes."""
    # Solo la estructura, los valores no se usan realmente
    return {
        "W1": np.zeros((input_size, hidden_size)),
        "b1": np.zeros(hidden_size),
        "W2": np.zeros((hidden_size, output_size)),
        "b2": np.zeros(output_size)
    }

def relu(Z: np.ndarray) -> np.ndarray:
    """Función de activación ReLU."""
    return np.maximum(0, Z)

def softmax(Z: np.ndarray) -> np.ndarray:
    """Softmax estable numéricamente."""
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def forward(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """Forward pass de la red."""
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    
    return softmax(Z2)

def backward(X: np.ndarray, y: np.ndarray, 
             weights: Dict[str, np.ndarray], 
             output: np.ndarray) -> Dict[str, np.ndarray]:
    """Backward pass para calcular gradientes."""
    m = X.shape[0]
    
    # Gradiente de la capa de salida
    dZ2 = output
    dZ2[np.arange(m), y] -= 1
    dZ2 = dZ2 / m
    
    # Gradientes para W2 y b2
    A1 = relu(X @ weights["W1"] + weights["b1"])
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)
    
    # Gradiente para capa oculta
    dA1 = dZ2 @ weights["W2"].T
    dZ1 = dA1 * (A1 > 0)  # Derivada de ReLU
    
    # Gradientes para W1 y b1
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)
    
    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

def train_epoch(X: np.ndarray, y: np.ndarray, 
                weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    """Entrena una época y retorna gradientes y loss."""
    # Forward pass
    output = forward(X, weights)
    
    # Calcular loss (cross-entropy)
    m = X.shape[0]
    correct_logprobs = -np.log(output[np.arange(m), y] + 1e-9)
    loss = np.sum(correct_logprobs) / m
    
    # Backward pass
    grads = backward(X, y, weights, output)
    
    return grads, loss