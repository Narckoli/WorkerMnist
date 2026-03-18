# dataset.py
import numpy as np
from torchvision import datasets, transforms
from typing import Tuple

def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Carga MNIST usando torchvision."""
    print("Cargando MNIST con torchvision...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Convertir a numpy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_train = train_dataset.targets.numpy()
    
    print(f"✓ Dataset cargado: {X_train.shape}")
    return X_train, y_train

def extract_chunk(X_train: np.ndarray, y_train: np.ndarray, indices: list) -> Tuple[np.ndarray, np.ndarray]:
    """Extrae un chunk del dataset según los índices recibidos."""
    indices_array = np.array(indices)
    X_chunk = X_train[indices_array]
    y_chunk = y_train[indices_array]
    return X_chunk, y_chunk