#!/usr/bin/env python3
"""
Entrenamiento centralizado (tradicional) de EfficientNetLite-0 en CIFAR-10.
Sin servidor, sin workers, sin comunicacion de red.
Para comparacion con entrenamiento distribuido.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Centralized")


# ==================== RESULTADOS ====================
@dataclass
class TrainingResult:
    training_id: str = ""
    start_time: str = ""
    end_time: str = ""
    num_workers: int = 1  # Centralizado = 1 "worker"
    num_classes: int = 10
    batch_size: int = 64
    max_epochs: int = 10
    learning_rate: float = 0.001
    dataset: str = "cifar10"
    total_epochs_completed: int = 0
    total_updates: int = 0
    final_loss: float = 0.0
    final_accuracy: float = 0.0
    total_seconds: float = 0.0
    avg_epoch_seconds: float = 0.0
    worker_hardware: dict = field(default_factory=dict)
    worker_updates: dict = field(default_factory=dict)
    epoch_history: list = field(default_factory=list)
    
    def to_dict(self):
        return asdict(self)
    
    def save(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path.home() / "training_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"training_centralized_{self.training_id}.json"
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Resultados guardados en: {filepath}")
        return filepath
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("RESUMEN DEL ENTRENAMIENTO CENTRALIZADO")
        print("=" * 70)
        print(f"   ID:              {self.training_id}")
        print(f"   Inicio:          {self.start_time}")
        print(f"   Fin:             {self.end_time}")
        print(f"   Duracion:        {self._format_time(self.total_seconds)}")
        print("-" * 70)
        print(f"   Dataset:         {self.dataset.upper()}")
        print(f"   Batch size:      {self.batch_size}")
        print(f"   Epocas:          {self.total_epochs_completed} / {self.max_epochs}")
        print(f"   Learning rate:   {self.learning_rate}")
        print("-" * 70)
        print(f"   Total updates:   {self.total_updates}")
        print(f"   Loss final:      {self.final_loss:.6f}")
        print(f"   Accuracy final:  {self.final_accuracy:.2f}%")
        print("-" * 70)
        print("   Hardware:")
        for wid, hw in self.worker_hardware.items():
            print(f"      {hw.get('cpu', 'Unknown')} ({hw.get('tier', '?')})")
        print("=" * 70)
    
    def _format_time(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs:.1f}s"


# ==================== MODELO (identico al distribuido) ====================
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_ch, se_ratio=0.25):
        super().__init__()
        se_ch = max(1, int(in_ch * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, se_ch, 1)
        self.fc2 = nn.Conv2d(se_ch, in_ch, 1)
        self.act = Swish()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        se = self.pool(x)
        se = self.act(self.fc1(se))
        se = self.sigmoid(self.fc2(se))
        return x * se

class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio, kernel_size, stride, se_ratio, drop_rate=0.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, hidden_dim, 1, bias=False), nn.BatchNorm2d(hidden_dim), Swish()]
        layers += [nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
                   nn.BatchNorm2d(hidden_dim), Swish()]
        if se_ratio is not None:
            layers.append(SEBlock(hidden_dim, se_ratio))
        layers += [nn.Conv2d(hidden_dim, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)
        self.drop_rate = drop_rate
    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.drop_rate > 0 and self.training:
                out = nn.functional.dropout(out, p=self.drop_rate, training=True)
            out = out + x
        return out

class EfficientNetLite0(nn.Module):
    CONFIG = [(1,16,3,1,None,1),(6,24,3,2,None,2),(6,40,5,2,None,2),(6,80,3,2,0.25,3),(6,112,5,1,0.25,3),(6,192,5,2,0.25,4),(6,320,3,1,None,1)]
    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super().__init__()
        out_ch = int(__import__('math').ceil(32 * width_mult / 8) * 8)
        self.stem = nn.Sequential(nn.Conv2d(3, out_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(out_ch), Swish())
        blocks, in_ch = [], out_ch
        for expand_ratio, out_ch_cfg, kernel_size, stride, se_ratio, num_repeat in self.CONFIG:
            out_ch = int(__import__('math').ceil(out_ch_cfg * width_mult / 8) * 8)
            num_repeat = int(__import__('math').ceil(num_repeat * depth_mult))
            for i in range(num_repeat):
                blocks.append(MBConvBlock(in_ch, out_ch, expand_ratio, kernel_size, stride if i==0 else 1, se_ratio, dropout_rate))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        head_ch = int(__import__('math').ceil(1280 * width_mult / 8) * 8)
        self.head = nn.Sequential(nn.Conv2d(in_ch, head_ch, 1, bias=False), nn.BatchNorm2d(head_ch), Swish(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(head_ch, num_classes))
    def forward(self, x):
        x = self.stem(x); x = self.blocks(x); x = self.head(x); x = x.view(x.size(0), -1); x = self.classifier(x); return x


# ==================== ENTRENAMIENTO CENTRALIZADO ====================
class CentralizedTrainer:
    def __init__(self, num_classes=10, batch_size=64, lr=0.001, 
                 max_epochs=10, dataset='cifar10', data_dir='./data',
                 save_results=True, results_dir=None):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs = max_epochs
        self.dataset = dataset
        self.data_dir = data_dir
        self.save_results = save_results
        
        if results_dir:
            self.results_dir = Path(results_dir)
        else:
            self.results_dir = Path(__file__).parent.resolve() / "Resultados"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = EfficientNetLite0(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        self.result = TrainingResult()
        self.result.training_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result.num_classes = num_classes
        self.result.batch_size = batch_size
        self.result.max_epochs = max_epochs
        self.result.learning_rate = lr
        self.result.dataset = dataset
        
        # Hardware info
        import platform, psutil
        cpu_info = platform.processor() or "Unknown"
        self.result.worker_hardware[0] = {
            "cpu": cpu_info,
            "tier": "centralized",
            "has_cuda": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

    def load_data(self):
        """Carga CIFAR-10 completo (50000 train, 10000 test)."""
        logger.info("Cargando CIFAR-10...")
        
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=transform_test
        )
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, pin_memory=False  # num_workers=0 para estabilidad
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=64, 
            shuffle=False, num_workers=0
        )
        
        logger.info(f"Datos listos: {len(train_dataset)} train, {len(test_dataset)} test")
        logger.info(f"Batches por epoca: {len(self.train_loader)}")

    def train_epoch(self, epoch):
        """Entrena una epoca completa. Retorna loss promedio."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        logger.info(f"Epoca {epoch}/{self.max_epochs} iniciada")
        epoch_start = time.time()
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Imprimir cada 20 batches (igual que worker distribuido)
            if batch_idx % 20 == 0 or batch_idx == len(self.train_loader) - 1:
                avg_so_far = epoch_loss / max(num_batches, 1)
                logger.info(f"   Batch {batch_idx}/{len(self.train_loader)} | "
                           f"Loss: {loss.item():.4f} | Media: {avg_so_far:.4f} | "
                           f"Progreso: {100*num_batches/len(self.train_loader):.1f}%")
        
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_duration = time.time() - epoch_start
        
        logger.info(f"Epoca {epoch} completada | Loss: {avg_loss:.4f} | Tiempo: {epoch_duration:.1f}s")
        
        return avg_loss, epoch_duration

    def evaluate(self):
        """Evalua accuracy en test set."""
        self.model.eval()
        correct = 0
        total = 0
        
        logger.info("Evaluando en test set...")
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
        return accuracy

    def train(self):
        """Entrenamiento completo."""
        self.load_data()
        
        logger.info("=" * 70)
        logger.info("ENTRENAMIENTO CENTRALIZADO INICIANDO")
        logger.info(f"   Modelo: EfficientNetLite-0")
        logger.info(f"   Dataset: {self.dataset.upper()}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Epocas: {self.max_epochs}")
        logger.info(f"   LR: {self.lr}")
        logger.info("=" * 70)
        
        start_time = time.time()
        self.result.start_time = datetime.now().isoformat()
        
        for epoch in range(self.max_epochs):
            avg_loss, epoch_duration = self.train_epoch(epoch)
            
            self.result.epoch_history.append({
                'epoch': epoch,
                'avg_loss': round(avg_loss, 6),
                'duration_seconds': round(epoch_duration, 2),
                'timestamp': datetime.now().isoformat()
            })
            self.result.total_updates += 1
        
        # Evaluacion final
        final_accuracy = self.evaluate()
        final_loss = self.result.epoch_history[-1]['avg_loss'] if self.result.epoch_history else 0.0
        
        end_time = time.time()
        total_seconds = end_time - start_time
        
        self.result.end_time = datetime.now().isoformat()
        self.result.total_seconds = round(total_seconds, 2)
        self.result.total_epochs_completed = len(self.result.epoch_history)
        self.result.final_loss = round(final_loss, 6)
        self.result.final_accuracy = round(final_accuracy, 2)
        
        if self.result.epoch_history:
            total_epoch_time = sum(e.get('duration_seconds', 0) for e in self.result.epoch_history)
            self.result.avg_epoch_seconds = round(total_epoch_time / len(self.result.epoch_history), 2)
        
        self.result.print_summary()
        
        if self.save_results:
            self.result.save(self.results_dir)
        
        return self.result


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento Centralizado EfficientNetLite-0')
    parser.add_argument('--num-classes', type=int, default=10, help='Clases')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Epocas')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'synthetic'])
    parser.add_argument('--data-dir', default='./data', help='Directorio de datos')
    parser.add_argument('--no-save', action='store_true', help='No guardar resultados')
    parser.add_argument('--results-dir', type=str, help='Directorio para resultados')
    
    args = parser.parse_args()
    
    trainer = CentralizedTrainer(
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.epochs,
        dataset=args.dataset,
        data_dir=args.data_dir,
        save_results=not args.no_save,
        results_dir=args.results_dir
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Entrenamiento interrumpido por usuario")