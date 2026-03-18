# config.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class WorkerConfig:
    """Configuración del worker."""
    # Conexión al servidor
    SERVER_IP: str = "192.168.61.230"
    PORT: int = 5000
    
    # Estado del worker
    worker_id: Optional[int] = None
    connected: bool = False
    
    # Datos locales
    X_chunk: Optional[np.ndarray] = None
    y_chunk: Optional[np.ndarray] = None
    
    @classmethod
    def from_input(cls):
        """Crea configuración preguntando al usuario."""
        ip = input(f"IP del servidor (default: {cls.SERVER_IP}): ").strip()
        if ip:
            cls.SERVER_IP = ip
        
        port = input(f"Puerto (default: {cls.PORT}): ").strip()
        if port:
            cls.PORT = int(port)
        
        return cls()

# Instancia global
config = WorkerConfig()