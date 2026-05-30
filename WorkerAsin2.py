"""
Worker 100% ASINCRONO.
- Entrena continuamente sin esperar a nadie
- Envía gradientes cada N batches (no al final de epoca)
- Recibe modelo actualizado inmediatamente
- NUNCA se bloquea
- Maneja version tracking para staleness
"""

import asyncio
import struct
import pickle
import logging
import argparse
import random
import platform
import psutil
import os
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("WorkerAsync")


# ==================== CONFIGURACION ====================
CONFIG_FILE = Path.home() / ".dist_train_config.json"
ENV_VAR_HOST = "DIST_SERVER_HOST"
ENV_VAR_PORT = "DIST_SERVER_PORT"
DEFAULT_HOST = "192.168.0.11"
DEFAULT_PORT = 5000


def load_config_file():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config.get('server_host'), config.get('server_port')
        except Exception:
            pass
    return None, None


def save_config_file(host, port):
    try:
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
        config['server_host'] = host
        config['server_port'] = port
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass


def get_env_config():
    host = os.environ.get(ENV_VAR_HOST)
    port = os.environ.get(ENV_VAR_PORT)
    if host:
        return host, int(port) if port else DEFAULT_PORT
    return None, None


def discover_server_on_network(port=5000, timeout=2.0):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        subnet = ".".join(local_ip.split(".")[:3])
    except Exception:
        return None, None
    
    for last_octet in [1, 10, 100, 2, 5, 254, 50, 20, 30] + list(range(1, 255)):
        host = f"{subnet}.{last_octet}"
        if host == local_ip:
            continue
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            if s.connect_ex((host, port)) == 0:
                s.close()
                save_config_file(host, port)
                return host, port
            s.close()
        except:
            pass
    return None, None


def resolve_server_config(args_host, args_port, auto_discover=False):
    if args_host and args_host != DEFAULT_HOST:
        save_config_file(args_host, args_port)
        return args_host, args_port
    
    env_host, env_port = get_env_config()
    if env_host:
        save_config_file(env_host, env_port)
        return env_host, env_port
    
    file_host, file_port = load_config_file()
    if file_host:
        return file_host, file_port or DEFAULT_PORT
    
    if auto_discover:
        discovered = discover_server_on_network(DEFAULT_PORT)
        if discovered[0]:
            return discovered
    
    return DEFAULT_HOST, DEFAULT_PORT


# ==================== HARDWARE ====================
def detect_hardware():
    cpu_info = platform.processor() or "Unknown"
    cpu_count = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().max if psutil.cpu_freq() else 0
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if "i9" in cpu_info or cpu_count >= 16:
        tier = "high"
    elif "i7" in cpu_info or "Ryzen 7" in cpu_info or cpu_count >= 12:
        tier = "medium"
    else:
        tier = "low"
    
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


# ==================== MODELO ====================
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


# ==================== PROTOCOLO ====================
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


# ==================== DATASET ====================
class AsyncDataset:
    """
    Dataset para entrenamiento asincrono.
    - Divide el dataset en chunks
    - Cada worker recibe asignacion de chunks
    - Shuffling por epoca para variedad
    """
    def __init__(self, dataset_name='cifar10', data_dir='./data', batch_size=32, 
                 chunk_size=10, worker_id=0, total_workers=1):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.worker_id = worker_id
        self.total_workers = total_workers
        
        logger.info("Cargando dataset...")
        self.full_dataset = self._load_dataset()
        self.total_samples = len(self.full_dataset)
        self.total_batches = (self.total_samples + batch_size - 1) // batch_size
        self.total_chunks = (self.total_batches + self.chunk_size - 1) // self.chunk_size
        
        # Asignar chunks a este worker
        self._assign_chunks()
        
        logger.info(f"Dataset listo: {self.total_samples} muestras | "
                   f"{self.total_batches} batches | {self.total_chunks} chunks | "
                   f"Worker {worker_id}: {len(self.assigned_chunks)} chunks")
    
    def _load_dataset(self):
        if self.dataset_name == 'cifar10':
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            return torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        else:
            X = torch.randn(10000, 3, 224, 224)
            y = torch.randint(0, 10, (10000,))
            from torch.utils.data import TensorDataset
            return TensorDataset(X, y)
    
    def _assign_chunks(self):
        """Asigna chunks a este worker de forma round-robin."""
        self.assigned_chunks = []
        for chunk_idx in range(self.total_chunks):
            if chunk_idx % self.total_workers == self.worker_id:
                self.assigned_chunks.append(chunk_idx)
    
    def shuffle_chunks(self, seed=None):
        """Reordena los chunks para la siguiente epoca."""
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.assigned_chunks)
    
    def get_sample_indices_for_chunks(self, chunk_indices):
        indices = []
        for chunk_idx in sorted(chunk_indices):
            start_batch = chunk_idx * self.chunk_size
            end_batch = min(start_batch + self.chunk_size, self.total_batches)
            start_sample = start_batch * self.batch_size
            end_sample = min(end_batch * self.batch_size, self.total_samples)
            indices.extend(range(start_sample, end_sample))
        return indices
    
    def create_dataloader(self, chunk_indices=None, num_workers=0):
        if chunk_indices is None:
            chunk_indices = self.assigned_chunks
        sample_indices = self.get_sample_indices_for_chunks(chunk_indices)
        subset = Subset(self.full_dataset, sample_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=False, drop_last=False)
    
    def get_all_batches_iterator(self):
        """Generador infinito de batches para entrenamiento continuo."""
        epoch = 0
        while True:
            self.shuffle_chunks(seed=42 + epoch)
            loader = self.create_dataloader()
            for batch in loader:
                yield batch, epoch
            epoch += 1


# ==================== WORKER ASINCRONO ====================
class AsyncWorker:
    """
    Worker completamente asincrono:
    - Entrena continuamente sin esperar
    - Acumula gradientes localmente
    - Envía al servidor cada N batches (gradient_frequency)
    - Recibe modelo actualizado inmediatamente
    - NUNCA se bloquea esperando a otros workers
    """
    
    def __init__(self, server_host='localhost', server_port=5000, num_classes=10, 
                 data_dir='./data', gradient_frequency=5):
        self.server_host = server_host
        self.server_port = server_port
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.gradient_frequency = gradient_frequency  # Enviar gradientes cada N batches
        
        self.worker_id = None
        self.total_workers = None
        self.model = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.hardware_info = detect_hardware()
        self.dataset = None
        self.max_updates = None
        
        # Version tracking
        self.local_model_version = 0
        self.server_model_version = 0
        
        # Acumuladores de gradientes
        self.accumulated_gradients = {}
        self.accumulated_loss = 0.0
        self.accumulated_count = 0
        
        # Estadisticas
        self.total_batches_processed = 0
        self.total_gradients_sent = 0
        self.staleness_history = []
        
        logger.info("=" * 60)
        logger.info("WORKER 100% ASINCRONO")
        logger.info(f"   Hardware: {self.hardware_info['cpu']}")
        logger.info(f"   Tier: {self.hardware_info['tier']}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Gradient frequency: cada {gradient_frequency} batches")
        logger.info("=" * 60)

    def create_model(self, state_dict=None):
        """Crea modelo. Si hay state_dict, carga SOLO pesos entrenables."""
        self.model = EfficientNetLite0(num_classes=self.num_classes).to(self.device)
        
        if state_dict is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in state_dict:
                        param.copy_(state_dict[name])
            logger.info(f"Modelo cargado ({len([n for n in state_dict.keys()])} parametros)")
        
        self.criterion = nn.CrossEntropyLoss()
        logger.info(f"Modelo listo en {self.device}")

    def zero_accumulated_gradients(self):
        """Limpia acumuladores de gradientes."""
        self.accumulated_gradients = {}
        self.accumulated_loss = 0.0
        self.accumulated_count = 0

    def accumulate_batch_gradient(self, data, target):
        """
        Calcula gradientes para un batch y los acumula localmente.
        NO aplica optimizer.step() - eso lo hace el servidor.
        """
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)
        
        # Forward + backward
        self.model.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        
        # Acumular gradientes
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_cpu = param.grad.cpu().clone()
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = grad_cpu
                else:
                    self.accumulated_gradients[name] += grad_cpu
        
        self.accumulated_loss += loss.item()
        self.accumulated_count += 1
        self.total_batches_processed += 1
        
        return loss.item()

    def get_averaged_gradients(self):
        """
        Promedia los gradientes acumulados.
        """
        if self.accumulated_count == 0:
            return {}
        
        averaged = {}
        for name, grad in self.accumulated_gradients.items():
            averaged[name] = grad / self.accumulated_count
        
        return averaged

    async def send_gradient_and_wait_model(self, writer, reader):
        """
        Envia gradientes acumulados al servidor y espera el modelo actualizado.
        """
        if self.accumulated_count == 0:
            return True
        
        gradients = self.get_averaged_gradients()
        avg_loss = self.accumulated_loss / self.accumulated_count
        
        logger.info(f"Enviando gradiente (batches: {self.accumulated_count}, "
                   f"loss: {avg_loss:.4f}, version local: {self.local_model_version})")
        
        await send_msg(writer, {
            'type': 'gradient',
            'gradients': gradients,
            'model_version': self.local_model_version,
            'loss': avg_loss,
            'worker_id': self.worker_id,
            'batch_count': self.accumulated_count
        })
        
        # Esperar confirmacion del servidor
        try:
            response = await asyncio.wait_for(recv_msg(reader), timeout=10.0)
            
            if response['type'] == 'gradient_received':
                logger.info(f"Gradiente aceptado | Queue: {response.get('queue_size', '?')} | "
                           f"Server version: {response.get('model_version', '?')}")
                
                # Ahora esperar el modelo actualizado
                model_msg = await asyncio.wait_for(recv_msg(reader), timeout=30.0)
                
                if model_msg['type'] == 'model_update':
                    new_version = model_msg['model_version']
                    staleness = new_version - self.local_model_version
                    self.staleness_history.append(staleness)
                    
                    logger.info(f"Modelo actualizado: v{self.local_model_version} -> v{new_version} "
                               f"(staleness: {staleness})")
                    
                    # Cargar nuevo modelo
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if name in model_msg['state_dict']:
                                param.copy_(model_msg['state_dict'][name])
                    
                    self.local_model_version = new_version
                    self.server_model_version = new_version
                    self.total_gradients_sent += 1
                    
                    # Limpiar acumuladores
                    self.zero_accumulated_gradients()
                    
                    # Verificar si terminamos
                    if model_msg.get('total_updates', 0) >= model_msg.get('max_updates', float('inf')):
                        logger.info("Max updates alcanzado por el servidor!")
                        return False
                    
                    return True
                    
                elif model_msg['type'] == 'training_done':
                    logger.info("Entrenamiento finalizado por el servidor!")
                    return False
                    
            elif response['type'] == 'gradient_rejected':
                logger.warning(f"Gradiente rechazado: {response.get('reason', 'unknown')}")
                self.zero_accumulated_gradients()
                return True
                
        except asyncio.TimeoutError:
            logger.warning("Timeout esperando respuesta del servidor")
            return True
        
        return True

    async def run(self):
        logger.info(f"Conectando a {self.server_host}:{self.server_port}...")
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.server_host, self.server_port),
                timeout=10.0
            )
            logger.info("Conectado al servidor!")
        except Exception as e:
            logger.error(f"Error de conexion: {e}")
            return
        
        try:
            # 1. Handshake
            await send_msg(writer, {'type': 'handshake', 'hardware': self.hardware_info})
            
            # 2. Recibir configuracion
            response = await recv_msg(reader)
            if response['type'] == 'assign_id':
                self.worker_id = response['worker_id']
                self.total_workers = response['total_workers']
                server_batch_size = response['batch_size']
                dataset_name = response['dataset']
                self.max_updates = response.get('max_updates', 1000)
                self.server_model_version = response.get('model_version', 0)
                
                logger.info(f"Asignado: Worker {self.worker_id}/{self.total_workers}")
                logger.info(f"   Batch size: {server_batch_size}")
                logger.info(f"   Max updates: {self.max_updates}")
            
            # 3. Cargar dataset
            self.dataset = AsyncDataset(
                dataset_name=dataset_name,
                data_dir=self.data_dir,
                batch_size=server_batch_size,
                chunk_size=10,
                worker_id=self.worker_id,
                total_workers=self.total_workers
            )
            
            # 4. Esperar modelo inicial
            logger.info("Esperando modelo inicial...")
            msg = await recv_msg(reader)
            
            if msg['type'] == 'model_update':
                self.create_model(msg['state_dict'])
                self.local_model_version = msg['model_version']
                logger.info(f"Modelo inicial recibido (version {self.local_model_version})")
            else:
                raise ValueError(f"Mensaje inesperado: {msg['type']}")
            
            # 5. Bucle de entrenamiento asincrono INFINITO
            logger.info("=" * 60)
            logger.info("INICIANDO ENTRENAMIENTO ASINCRONO CONTINUO")
            logger.info("=" * 60)
            
            batch_iterator = self.dataset.get_all_batches_iterator()
            self.zero_accumulated_gradients()
            
            running = True
            while running:
                # Entrenar batches y acumular gradientes
                for _ in range(self.gradient_frequency):
                    try:
                        (data, target), epoch = next(batch_iterator)
                        loss = self.accumulate_batch_gradient(data, target)
                        
                        # Log cada 20 batches
                        if self.total_batches_processed % 20 == 0:
                            avg_loss = self.accumulated_loss / max(self.accumulated_count, 1)
                            logger.info(f"Batch {self.total_batches_processed} | "
                                       f"Loss: {loss:.4f} | Acum loss: {avg_loss:.4f} | "
                                       f"Version: {self.local_model_version}")
                        
                    except StopIteration:
                        break
                    except Exception as e:
                        logger.error(f"Error en batch: {e}")
                        continue
                
                # Enviar gradientes acumulados al servidor
                if self.accumulated_count > 0:
                    running = await self.send_gradient_and_wait_model(writer, reader)
                
                # Verificar si el servidor indico fin
                # (checkear mensajes pendientes sin bloquear)
                try:
                    pending_msg = await asyncio.wait_for(recv_msg(reader), timeout=0.1)
                    if pending_msg['type'] == 'training_done':
                        logger.info("Recibido training_done del servidor")
                        running = False
                        break
                    elif pending_msg['type'] == 'heartbeat':
                        # Responder heartbeat
                        pass
                except asyncio.TimeoutError:
                    pass
            
            # Finalizar
            await send_msg(writer, {'type': 'done'})
            
            # Estadisticas finales
            avg_staleness = sum(self.staleness_history) / len(self.staleness_history) if self.staleness_history else 0
            logger.info("=" * 60)
            logger.info("WORKER FINALIZADO")
            logger.info(f"   Batches procesados: {self.total_batches_processed}")
            logger.info(f"   Gradientes enviados: {self.total_gradients_sent}")
            logger.info(f"   Staleness promedio: {avg_staleness:.2f}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error en Worker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except:
                pass


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker 100% Asincrono',
        epilog="Ejemplo: python WorkerAsync.py --server-host 192.168.0.11"
    )
    parser.add_argument('--server-host', default=DEFAULT_HOST)
    parser.add_argument('--server-port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--auto-discover', action='store_true')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--gradient-frequency', type=int, default=5,
                        help='Enviar gradientes al servidor cada N batches')
    
    args = parser.parse_args()
    
    host, port = resolve_server_config(args.server_host, args.server_port, args.auto_discover)
    
    worker = AsyncWorker(
        server_host=host, 
        server_port=port, 
        num_classes=args.num_classes, 
        data_dir=args.data_dir,
        gradient_frequency=args.gradient_frequency
    )
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Worker detenido")
