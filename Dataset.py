# dataset.py (worker)
import numpy as np
from torchvision import datasets, transforms
from typing import Tuple, List

def load_mnist_chunk(indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Carga un chunk específico de MNIST."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Extraer solo los índices solicitados
    X = train_dataset.data.numpy()[indices].reshape(-1, 784) / 255.0
    y = train_dataset.targets.numpy()[indices]
    
    return X, y

def load_cifar10_chunk(indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Carga un chunk específico de CIFAR-10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Extraer solo los índices solicitados
    X = train_dataset.data[indices].reshape(-1, 3072) / 255.0
    y = np.array(train_dataset.targets)[indices]
    
    return X, y

def load_dataset_chunk(dataset_name: str, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Carga un chunk del dataset especificado."""
    print(f"Cargando chunk de {dataset_name.upper()} con {len(indices)} muestras...")
    
    if dataset_name.lower() == 'mnist':
        X, y = load_mnist_chunk(indices)
    elif dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar-10':
        X, y = load_cifar10_chunk(indices)
    else:
        raise ValueError(f"Dataset '{dataset_name}' no soportado")
    
    print(f"✓ Chunk cargado: X shape={X.shape}, y shape={y.shape}")
    return X, y