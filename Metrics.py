# metrics.py
import numpy as np
from typing import List, Dict

class LocalMetrics:
    """Métricas locales del worker."""
    
    def __init__(self):
        self.epoch_losses: List[float] = []
        self.epoch_times: List[float] = []
        self.current_epoch = 0
    
    def add_epoch_result(self, loss: float, time_taken: float):
        """Añade resultado de una época."""
        self.epoch_losses.append(loss)
        self.epoch_times.append(time_taken)
        self.current_epoch += 1
    
    def print_summary(self):
        """Imprime resumen del entrenamiento local."""
        if not self.epoch_losses:
            return
        
        print("\n" + "="*50)
        print("RESUMEN LOCAL DEL WORKER")
        print("="*50)
        print(f"Épocas completadas: {len(self.epoch_losses)}")
        print(f"Loss final: {self.epoch_losses[-1]:.4f}")
        print(f"Loss promedio: {np.mean(self.epoch_losses):.4f}")
        print(f"Mejor loss: {np.min(self.epoch_losses):.4f}")
        print(f"Tiempo promedio por época: {np.mean(self.epoch_times):.2f}s")
        print("="*50)

# Instancia global de métricas
metrics = LocalMetrics()