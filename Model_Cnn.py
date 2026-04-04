# model_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

# Configuración global del modelo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Modelo CNN usando: {device}")

class CIFAR10CNN(nn.Module):
    """
    Red Neuronal Convolucional para CIFAR-10
    Arquitectura inspirada en VGG simplificada
    """
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # Bloque Convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Bloque Convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Bloque Convolucional 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Capas completamente conectadas
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # 4x4 después de pooling
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.5)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Bloque 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Bloque 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Bloque 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Aplanar para capas fully connected
        x = x.view(-1, 128 * 4 * 4)
        
        # Capas fully connected con dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class ModelCNN:
    """
    Wrapper para usar el modelo CNN en el servidor
    Compatible con la interfaz existente
    """
    def __init__(self, input_shape=(3, 32, 32), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.device = device
        self.model = None
        self.is_initialized = False
        
    def init_model(self):
        """Inicializa el modelo CNN"""
        self.model = CIFAR10CNN(num_classes=self.num_classes).to(self.device)
        self.is_initialized = True
        print(f"✅ Modelo CNN inicializado en {self.device}")
        print(f"   Arquitectura:")
        print(f"   - Conv1: 3→32 (3x3)")
        print(f"   - Conv2: 32→64 (3x3)")
        print(f"   - Conv3: 64→128 (3x3)")
        print(f"   - FC1: 2048→256")
        print(f"   - FC2: 256→10")
        print(f"   - Total parámetros: {self.count_parameters():,}")
        
    def count_parameters(self) -> int:
        """Cuenta el número total de parámetros entrenables"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def forward_numpy(self, X: np.ndarray, weights: Dict = None) -> np.ndarray:
        """
        Forward pass usando numpy arrays (compatible con interfaz existente)
        NOTA: Esta función es para compatibilidad, pero CNN funciona mejor con PyTorch
        """
        if not self.is_initialized:
            self.init_model()
        
        # Convertir numpy a torch tensor
        # X shape: (batch_size, 3072) o (batch_size, 3, 32, 32)
        if X.shape[1] == 3072:  # Si es formato aplanado de CIFAR-10
            X = X.reshape(-1, 3, 32, 32)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        return probabilities.cpu().numpy()
    
    def get_weights(self) -> Dict:
        """Obtiene los pesos del modelo como diccionario de numpy arrays"""
        if not self.is_initialized:
            self.init_model()
        
        weights = {}
        for name, param in self.model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        
        return weights
    
    def set_weights(self, weights: Dict):
        """Establece los pesos del modelo desde un diccionario"""
        if not self.is_initialized:
            self.init_model()
        
        for name, param in self.model.named_parameters():
            if name in weights:
                param.data = torch.FloatTensor(weights[name]).to(self.device)
    
    def compute_loss_and_gradients(self, X: np.ndarray, y: np.ndarray, 
                                   weights: Dict, loss_fn=nn.CrossEntropyLoss()) -> Tuple[Dict, float]:
        """
        Calcula gradientes y pérdida para un batch
        Esta función se usará en los workers
        """
        if not self.is_initialized:
            self.init_model()
        
        # Establecer pesos
        self.set_weights(weights)
        self.model.train()
        
        # Convertir a tensores
        if X.shape[1] == 3072:  # Si es formato aplanado
            X = X.reshape(-1, 3, 32, 32)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Forward pass
        outputs = self.model(X_tensor)
        loss = loss_fn(outputs, y_tensor)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Obtener gradientes
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy()
        
        return gradients, loss.item()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Evalúa el modelo en un conjunto de datos"""
        if not self.is_initialized:
            self.init_model()
        
        self.model.eval()
        
        # Convertir a tensores
        if X.shape[1] == 3072:
            X = X.reshape(-1, 3, 32, 32)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = F.cross_entropy(outputs, y_tensor)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_tensor).float().mean()
        
        return loss.item(), accuracy.item()
    
    def init_weights(self, input_size: int = None) -> Dict:
        """Inicializa pesos (compatible con interfaz existente)"""
        if not self.is_initialized:
            self.init_model()
        
        # La inicialización ya se hizo en __init__ del modelo
        return self.get_weights()
    
    def average_gradients(self, all_grads: list) -> Dict:
        """Promedia gradientes de múltiples workers"""
        if not all_grads:
            return {}
        
        avg_grads = {}
        for key in all_grads[0].keys():
            avg_grads[key] = np.mean([grads[key] for grads in all_grads], axis=0)
        
        return avg_grads
    
    def apply_gradients(self, weights: Dict, grads: Dict, lr: float) -> Dict:
        """Aplica gradientes a los pesos"""
        new_weights = {}
        for key in weights.keys():
            new_weights[key] = weights[key] - lr * grads[key]
        
        return new_weights

# Función de conveniencia para crear el modelo
def create_cnn_model():
    """Crea y retorna una instancia del modelo CNN"""
    return ModelCNN()