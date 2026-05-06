
# ============================================================
# WORKER V2 - Con mezcla de batches y partición inteligente
# ============================================================
"""
Worker de entrenamiento distribuido asíncrono para EfficientNetLite-0.

Diseño para hardware heterogéneo:
- AMD Ryzen 3 Serie 7000 (4C/8T): batch_size=16-32, workers=0 (evitar overhead)
- Intel Core i9 10th Gen (10C/20T): batch_size=64-128, workers=4

Características:
- Recibe worker_id + total_workers del servidor
- Recibe asignación de CHUNKS (no índices fijos) para cada época
- Convierte chunks a índices de muestra del dataset
- Al cambiar de época, recibe NUEVA asignación → ve datos diferentes
- Todos los workers ven TODO el dataset progresivamente
"""

import asyncio
import struct
import pickle
import logging
import argparse
import random
import platform
import psutil
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ==================== CONFIGURACIÓN DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Worker")


# ==================== DETECCIÓN DE HARDWARE ====================
def detect_hardware():
    """
    Detecta el hardware del worker para ajustar configuración.
    
    Returns:
        dict con info del hardware para que el servidor lo sepa
    """
    cpu_info = platform.processor() or "Unknown"
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    # Clasificación simple
    if "i9" in cpu_info or cpu_count >= 16:
        tier = "high"
    elif "i7" in cpu_info or "Ryzen 7" in cpu_info or cpu_count >= 12:
        tier = "medium"
    else:
        tier = "low"  # Ryzen 3, i3, etc.
    
    return {
        "cpu": cpu_info,
        "cores_logical": cpu_count,
        "cores_physical": psutil.cpu_count(logical=False),
        "cpu_freq_mhz": cpu_freq,
        "ram_gb": round(ram_gb, 1),
        "tier": tier,
        "has_cuda": torch.cuda.is_available(),
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


def get_optimal_config(hardware_info):
    """
    Devuelve configuración óptima según el hardware detectado.
    
    Ryzen 3 7000 series (4C/8T):
    - batch_size: 16-32 (evitar saturar RAM)
    - num_workers: 0 (asyncio ya maneja concurrencia, evitar overhead de multiprocessing)
    - prefetch: desactivado
    
    Core i9 10th Gen (10C/20T):
    - batch_size: 64-128
    - num_workers: 2-4
    - prefetch: activado
    """
    tier = hardware_info.get("tier", "low")
    
    if tier == "high":
        return {"batch_size": 64, "num_workers": 4, "prefetch": True}
    elif tier == "medium":
        return {"batch_size": 32, "num_workers": 2, "prefetch": True}
    else:  # low - Ryzen 3
        return {"batch_size": 16, "num_workers": 0, "prefetch": False}


# ==================== MODELO EFFICIENTNET LITE-0 ====================
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
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_residual = (stride == 1 and in_ch == out_ch)
        hidden_dim = in_ch * expand_ratio

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ]
        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ]
        if se_ratio is not None:
            layers.append(SEBlock(hidden_dim, se_ratio))
        layers += [
            nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ]
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
    CONFIG = [
        (1, 16, 3, 1, None, 1),
        (6, 24, 3, 2, None, 2),
        (6, 40, 5, 2, None, 2),
        (6, 80, 3, 2, 0.25, 3),
        (6, 112, 5, 1, 0.25, 3),
        (6, 192, 5, 2, 0.25, 4),
        (6, 320, 3, 1, None, 1),
    ]

    def __init__(self, num_classes=10, width_mult=1.0, depth_mult=1.0, dropout_rate=0.2):
        super().__init__()
        out_ch = self._round_filters(32, width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            Swish()
        )
        
        blocks = []
        in_ch = out_ch
        for expand_ratio, out_ch_cfg, kernel_size, stride, se_ratio, num_repeat in self.CONFIG:
            out_ch = self._round_filters(out_ch_cfg, width_mult)
            num_repeat = self._round_repeats(num_repeat, depth_mult)
            for i in range(num_repeat):
                s = stride if i == 0 else 1
                blocks.append(MBConvBlock(in_ch, out_ch, expand_ratio, kernel_size, s, se_ratio, dropout_rate))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        
        head_ch = self._round_filters(1280, width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            Swish(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(head_ch, num_classes)
        )

    def _round_filters(self, filters, mult):
        from math import ceil
        return int(ceil(filters * mult / 8) * 8)

    def _round_repeats(self, repeats, mult):
        from math import ceil
        return int(ceil(repeats * mult))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== PROTOCOLO DE COMUNICACIÓN ====================
async def send_msg(writer, data):
    payload = pickle.dumps(data)
    length = struct.pack('>I', len(payload))
    writer.write(length + payload)
    await writer.drain()


async def recv_msg(reader):
    length_data = await reader.readexactly(4)
    length = struct.unpack('>I', length_data)[0]
    payload = await reader.readexactly(length)
    return pickle.loads(payload)


# ==================== DATASET CON CHUNKS ====================
class ChunkedDataset:
    """
    Gestiona el dataset CIFAR-10 con asignación por CHUNKS.
    
    ¿CÓMO FUNCIONA LA MEZCLA DE BATCHES?
    =======================================
    
    Problema original (SIN mezcla):
    - Worker 0 siempre ve muestras [0-24999]
    - Worker 1 siempre ve muestras [25000-49999]
    - NUNCA ven los datos del otro → modelo no generaliza bien
    
    Solución (CON mezcla de chunks):
    - El dataset se divide en "chunks" (grupos de batches consecutivos)
    - Cada época, el servidor REASIGNA chunks aleatoriamente
    - Worker 0 en época 0: chunks [0, 2, 4] → muestras [0-319], [640-959], ...
    - Worker 0 en época 1: chunks [1, 3, 5] → muestras [320-639], [960-1279], ...
    - Worker 0 en época 2: chunks [7, 9, 11] → muestras diferentes otra vez
    
    Resultado: Después de N épocas, TODOS los workers han visto TODO el dataset.
    """
    
    def __init__(self, dataset_name='cifar10', data_dir='./data', 
                 batch_size=32, chunk_size=10):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # Cargar dataset completo
        self.full_dataset = self._load_dataset()
        self.total_samples = len(self.full_dataset)
        self.total_batches = (self.total_samples + batch_size - 1) // batch_size
        self.total_chunks = (self.total_batches + chunk_size - 1) // chunk_size
        
        logger.info(f"📊 Dataset cargado: {dataset_name}")
        logger.info(f"   Total muestras: {self.total_samples}")
        logger.info(f"   Total batches: {self.total_batches}")
        logger.info(f"   Total chunks: {self.total_chunks} (chunk_size={chunk_size})")
    
    def _load_dataset(self):
        """Carga CIFAR-10 o dataset sintético."""
        if self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            return torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=True, transform=transform
            )
        else:
            # Dataset sintético para pruebas
            logger.info("Usando dataset sintético")
            X = torch.randn(10000, 3, 224, 224)
            y = torch.randint(0, 10, (10000,))
            from torch.utils.data import TensorDataset
            return TensorDataset(X, y)
    
    def get_sample_indices_for_chunks(self, chunk_indices: list) -> list:
        """
        Convierte una lista de chunk_indices a índices de muestra.
        
        Ejemplo:
            chunk_indices = [0, 2]
            chunk_size = 10, batch_size = 32
            → chunk 0: batches [0-9] → muestras [0-319]
            → chunk 2: batches [20-29] → muestras [640-959]
            → return: [0, 1, 2, ..., 319, 640, 641, ..., 959]
        """
        indices = []
        for chunk_idx in sorted(chunk_indices):
            start_batch = chunk_idx * self.chunk_size
            end_batch = min(start_batch + self.chunk_size, self.total_batches)
            
            start_sample = start_batch * self.batch_size
            end_sample = min(end_batch * self.batch_size, self.total_samples)
            
            indices.extend(range(start_sample, end_sample))
        
        return indices
    
    def create_dataloader(self, chunk_indices: list, num_workers: int = 0) -> DataLoader:
        """
        Crea un DataLoader para los chunks asignados.
        
        Args:
            chunk_indices: Lista de índices de chunks asignados por el servidor
            num_workers: Número de workers del DataLoader (0 para Ryzen 3)
        
        Returns:
            DataLoader con las muestras correspondientes a los chunks
        """
        sample_indices = self.get_sample_indices_for_chunks(chunk_indices)
        
        subset = Subset(self.full_dataset, sample_indices)
        
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,  # Mezclar dentro de los chunks asignados
            num_workers=num_workers,
            pin_memory=False,  # False para CPU, True para GPU
            drop_last=False
        )
        
        logger.info(f"   DataLoader creado: {len(sample_indices)} muestras, "
                    f"{len(loader)} batches, chunks: {chunk_indices}")
        
        return loader


# ==================== WORKER ====================
class AsyncDistWorker:
    def __init__(self, server_host='10.253.18.128', server_port=5000, 
                 num_classes=10, data_dir='./data'):
        self.server_host = server_host
        self.server_port = server_port
        self.num_classes = num_classes
        self.data_dir = data_dir
        
        self.worker_id = None
        self.total_workers = None
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hardware detection
        self.hardware_info = detect_hardware()
        self.optimal_config = get_optimal_config(self.hardware_info)
        
        # Dataset manager (se inicializa después de recibir config del servidor)
        self.dataset = None
        self.chunked_dataset = None
        
        logger.info("=" * 60)
        logger.info("⚙️  WORKER INICIANDO")
        logger.info(f"   Hardware: {self.hardware_info['cpu']}")
        logger.info(f"   Tier: {self.hardware_info['tier']}")
        logger.info(f"   Cores: {self.hardware_info['cores_physical']}F/{self.hardware_info['cores_logical']}T")
        logger.info(f"   RAM: {self.hardware_info['ram_gb']} GB")
        logger.info(f"   CUDA: {self.hardware_info['has_cuda']}")
        logger.info(f"   Config óptima: batch={self.optimal_config['batch_size']}, "
                    f"workers={self.optimal_config['num_workers']}")
        logger.info("=" * 60)

    def create_model(self, state_dict=None):
        """Crea el modelo local, opcionalmente cargando pesos del servidor."""
        self.model = EfficientNetLite0(num_classes=self.num_classes).to(self.device)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        logger.info(f"🧠 Modelo EfficientNetLite-0 creado en {self.device}")

    def compute_gradients(self, data, target):
        """Calcula gradientes en un batch local."""
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)
        
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.cpu().clone()
        
        return gradients, loss.item()

    def update_model(self, state_dict):
        """Actualiza el modelo local con el estado global."""
        self.model.load_state_dict(state_dict)

    async def run(self):
        """Bucle principal del worker."""
        reader, writer = await asyncio.open_connection(self.server_host, self.server_port)
        
        try:
            # 1. Handshake con info de hardware
            await send_msg(writer, {
                'type': 'handshake',
                'hardware': self.hardware_info
            })
            
            # 2. Recibir ID asignado + configuración del servidor
            response = await recv_msg(reader)
            if response['type'] == 'assign_id':
                self.worker_id = response['worker_id']
                self.total_workers = response['total_workers']
                server_batch_size = response['batch_size']
                chunk_size = response['chunk_size']
                dataset_name = response['dataset']
                
                logger.info("=" * 60)
                logger.info(f"🆔 ASIGNACIÓN RECIBIDA")
                logger.info(f"   Worker ID: {self.worker_id} / {self.total_workers}")
                logger.info(f"   Dataset: {dataset_name}")
                logger.info(f"   Batch size (servidor): {server_batch_size}")
                logger.info(f"   Chunk size: {chunk_size}")
                logger.info("=" * 60)
            
            # 3. Cargar dataset
            self.chunked_dataset = ChunkedDataset(
                dataset_name=dataset_name,
                data_dir=self.data_dir,
                batch_size=server_batch_size,
                chunk_size=chunk_size
            )
            
            # 4. Esperar inicio de época 0 (modelo + asignación de chunks)
            logger.info("⏳ Esperando modelo inicial y asignación de chunks...")
            msg = await recv_msg(reader)
            
            if msg['type'] == 'epoch_start':
                epoch = msg['epoch']
                state_dict = msg['state_dict']
                chunk_assignment = msg['chunk_assignment']
                
                # Crear modelo con pesos iniciales del servidor
                self.create_model(state_dict)
                
                # Obtener mis chunks para esta época
                my_chunks = chunk_assignment.get(self.worker_id, [])
                logger.info(f"🎯 Época {epoch} | Chunks asignados: {my_chunks}")
                
                # Crear DataLoader con mis chunks
                train_loader = self.chunked_dataset.create_dataloader(
                    my_chunks,
                    num_workers=self.optimal_config['num_workers']
                )
            else:
                raise ValueError(f"Mensaje inesperado: {msg['type']}")
            
            # 5. Bucle de entrenamiento
            global_step = 0
            current_epoch = epoch
            max_epochs = 10  # Ajustable
            
            while current_epoch < max_epochs:
                epoch_loss = 0.0
                num_batches = 0
                
                logger.info(f"🏃 Worker {self.worker_id} | Iniciando época {current_epoch}")
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    # Calcular gradientes
                    gradients, loss = self.compute_gradients(data, target)
                    epoch_loss += loss
                    num_batches += 1
                    global_step += 1
                    
                    # Enviar gradientes al servidor (ASÍNCRONO)
                    await send_msg(writer, {
                        'type': 'gradients',
                        'gradients': gradients,
                        'step': global_step,
                        'loss': loss,
                        'worker_id': self.worker_id,
                        'epoch': current_epoch
                    })
                    
                    # Recibir modelo actualizado (ASÍNCRONO)
                    try:
                        msg = await asyncio.wait_for(recv_msg(reader), timeout=60.0)
                        
                        if msg['type'] == 'model_update':
                            self.update_model(msg['state_dict'])
                            
                        elif msg['type'] == 'epoch_start':
                            # ¡Nueva época! El servidor reasignó chunks
                            new_epoch = msg['epoch']
                            new_state_dict = msg['state_dict']
                            new_assignment = msg['chunk_assignment']
                            
                            self.update_model(new_state_dict)
                            current_epoch = new_epoch
                            
                            my_chunks = new_assignment.get(self.worker_id, [])
                            logger.info(f"🎲 NUEVA ÉPOCA {current_epoch} | "
                                        f"Nuevos chunks: {my_chunks}")
                            
                            # Crear NUEVO DataLoader con nuevos chunks
                            train_loader = self.chunked_dataset.create_dataloader(
                                my_chunks,
                                num_workers=self.optimal_config['num_workers']
                            )
                            break  # Salir del loop de batches para reiniciar con nuevo loader
                            
                    except asyncio.TimeoutError:
                        logger.warning("⏱️ Timeout esperando modelo. Continuando...")
                    
                    if batch_idx % 20 == 0:
                        logger.info(f"   Worker {self.worker_id} | Epoch {current_epoch} | "
                                    f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}")
                
                # Época completada
                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"📈 Worker {self.worker_id} | Época {current_epoch} COMPLETADA | "
                            f"Loss: {avg_loss:.4f} | Batches: {num_batches}")
                
                # Notificar al servidor que completé esta época
                await send_msg(writer, {
                    'type': 'epoch_complete',
                    'epoch': current_epoch,
                    'worker_id': self.worker_id,
                    'avg_loss': avg_loss
                })
                
                current_epoch += 1
            
            # Finalizar
            await send_msg(writer, {'type': 'done'})
            logger.info(f"🏁 Worker {self.worker_id} finalizó correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error en Worker: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            writer.close()
            await writer.wait_closed()


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker de entrenamiento distribuido asíncrono - EfficientNetLite-0'
    )
    parser.add_argument('--server-host', default='localhost', help='Host del servidor')
    parser.add_argument('--server-port', type=int, default=8765, help='Puerto del servidor')
    parser.add_argument('--num-classes', type=int, default=10, help='Número de clases')
    parser.add_argument('--data-dir', default='./data', help='Directorio de datos')
    
    args = parser.parse_args()
    
    worker = AsyncDistWorker(
        server_host=args.server_host,
        server_port=args.server_port,
        num_classes=args.num_classes,
        data_dir=args.data_dir
    )
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("🛑 Worker detenido por el usuario")
