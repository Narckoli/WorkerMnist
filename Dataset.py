# dataset.py
import numpy as np
from torchvision import datasets, transforms
from typing import Tuple, List

def load_mnist_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga MNIST usando torchvision."""
    print("Cargando MNIST con torchvision...")
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    X_train = train_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 784) / 255.0
    y_test = test_dataset.targets.numpy()
    
    print(f"✓ Dataset cargado: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, y_train, X_test, y_test

def load_cifar10_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga CIFAR-10 usando torchvision."""
    print("Cargando CIFAR-10 con torchvision...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # CIFAR-10: 32x32x3 = 3072 características
    X_train = train_dataset.data.reshape(-1, 3072) / 255.0
    y_train = np.array(train_dataset.targets)
    X_test = test_dataset.data.reshape(-1, 3072) / 255.0
    y_test = np.array(test_dataset.targets)
    
    print(f"✓ Dataset cargado: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"  Dimensiones de entrada: {X_train.shape[1]} características")
    return X_train, y_train, X_test, y_test

def load_dataset_by_name(dataset_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga el dataset según el nombre especificado."""
    if dataset_name.lower() == 'mnist':
        return load_mnist_dataset()
    elif dataset_name.lower() == 'cifar10' or dataset_name.lower() == 'cifar-10':
        return load_cifar10_dataset()
    else:
        raise ValueError(f"Dataset '{dataset_name}' no soportado. Opciones: mnist, cifar10")

def stratified_split(y: np.ndarray, n_workers: int) -> List[np.ndarray]:
    """Divide el dataset estratificadamente por clases."""
    class_indices = {label: np.where(y == label)[0] for label in range(10)}
    
    # Mezclar índices de cada clase
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    # Distribuir entre workers
    worker_chunks = [[] for _ in range(n_workers)]
    
    for indices in class_indices.values():
        splits = np.array_split(indices, n_workers)
        for i, split in enumerate(splits):
            worker_chunks[i].extend(split)
    
    # Mezclar cada chunk
    for i in range(n_workers):
        worker_chunks[i] = np.array(worker_chunks[i])
        np.random.shuffle(worker_chunks[i])
    
    return worker_chunks