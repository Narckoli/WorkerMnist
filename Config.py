# config.py
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class WorkerConfig:
    """Configuración del worker."""
    # Conexión al servidor
    SERVER_IP: str = "192.168.0.5"
    PORT: int = 5000
    
    # Estado del worker
    worker_id: Optional[int] = None
    connected: bool = False
    
    # Información del dataset
    dataset_name: Optional[str] = None
    input_size: Optional[int] = None
    hidden_size: Optional[int] = None
    output_size: Optional[int] = None
    
    # Datos locales
    X_chunk: Optional[np.ndarray] = None
    y_chunk: Optional[np.ndarray] = None
    
    # Template para gradientes
    local_weights_template: Optional[dict] = None
    
    def configure_from_input(self):
        """Configura la instancia actual preguntando al usuario."""
        ip = input(f"IP del servidor (default: {self.SERVER_IP}): ").strip()
        if ip:
            self.SERVER_IP = ip
        
        port = input(f"Puerto (default: {self.PORT}): ").strip()
        if port:
            self.PORT = int(port)
        
        return self
    
    @classmethod
    def from_input(cls):
        """Crea una nueva configuración preguntando al usuario."""
        config = cls()
        return config.configure_from_input()
    
    def print_info(self):
        """Muestra información de configuración actual."""
        print(f"\n{'='*50}")
        print(f"CONFIGURACIÓN DEL WORKER")
        print(f"{'='*50}")
        print(f"Worker ID: {self.worker_id}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Dimensiones: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        print(f"{'='*50}")

# Instancia global
config = WorkerConfig()