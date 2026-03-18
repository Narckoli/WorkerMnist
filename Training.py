# training.py
import numpy as np
from typing import Dict, Tuple

from Model import forward, compute_loss, backward

def train_epoch(X_chunk: np.ndarray, 
                y_chunk: np.ndarray, 
                weights: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Entrena una época con el chunk local.
    Retorna: (gradientes, loss)
    """
    # Forward
    A2, cache = forward(X_chunk, weights)
    
    # Calcular loss
    loss = compute_loss(A2, y_chunk)
    
    # Backward
    grads = backward(A2, y_chunk, cache, weights)
    
    return grads, loss

def verify_weights(weights: Dict[str, np.ndarray]) -> None:
    """Verifica que los pesos sean válidos (debug)."""
    total = sum(np.sum(np.abs(w)) for w in weights.values())
    print(f"[DEBUG] Suma total de pesos: {total:.6f}")