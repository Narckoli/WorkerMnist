#!/usr/bin/env python3
"""
Worker de entrenamiento distribuido asincrono para EfficientNetLite-0.
Compatible con servidor v4 (recibe max_epochs del servidor).
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
logger = logging.getLogger("Worker")


# ==================== CONFIGURACION DEL SERVIDOR ====================
CONFIG_FILE = Path.home() / ".dist_train_config.json"
ENV_VAR_HOST = "DIST_SERVER_HOST"
ENV_VAR_PORT = "DIST_SERVER_PORT"
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765


def load_config_file():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            logger.info(f"Config cargada desde {CONFIG_FILE}")
            return config.get('server_host'), config.get('server_port')
        except Exception as e:
            logger.warning(f"Error leyendo config: {e}")
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
        logger.info(f"Config guardada en {CONFIG_FILE}")
    except Exception as e:
        logger.warning(f"Error guardando config: {e}")


def get_env_config():
    host = os.environ.get(ENV_VAR_HOST)
    port = os.environ.get(ENV_VAR_PORT)
    if host:
        logger.info(f"Env var: {ENV_VAR_HOST}={host}")
        return host, int(port) if port else DEFAULT_PORT
    return None, None


def discover_server_on_network(port=8765, timeout=2.0):
    logger.info("Escaneando red local...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        subnet = ".".join(local_ip.split(".")[:3])
        logger.info(f"Subred: {subnet}.x (mi IP: {local_ip})")
    except Exception:
        logger.warning("No se pudo detectar subred")
        return None, None
    
    common_ips = [1, 10, 100, 2, 5, 254, 50, 20, 30]
    for last_octet in common_ips:
        host = f"{subnet}.{last_octet}"
        if host == local_ip:
            continue
        if _try_connect(host, port, timeout):
            logger.info(f"Servidor en {host}:{port}")
            return host, port
    
    logger.info("Escaneando subred completa...")
    for last_octet in range(1, 255):
        if last_octet in common_ips:
            continue
        host = f"{subnet}.{last_octet}"
        if host == local_ip:
            continue
        if _try_connect(host, port, timeout):
            logger.info(f"Servidor en {host}:{port}")
            return host, port
    
    logger.warning("No se encontro servidor")
    return None, None


def _try_connect(host, port, timeout):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        result = s.connect_ex((host, port))
        s.close()
        return result == 0
    except:
        return False


def resolve_server_config(args_host, args_port, auto_discover=False):
    logger.info("Resolviendo configuracion del servidor...")
    
    if args_host and args_host != DEFAULT_HOST:
        logger.info(f"1. Argumento: {args_host}:{args_port}")
        save_config_file(args_host, args_port)
        return args_host, args_port
    
    env_host, env_port = get_env_config()
    if env_host:
        logger.info(f"2. Env var: {env_host}:{env_port}")
        save_config_file(env_host, env_port)
        return env_host, env_port
    
    file_host, file_port = load_config_file()
    if file_host:
        logger.info(f"3. Config file: {file_host}:{file_port or DEFAULT_PORT}")
        return file_host, file_port or DEFAULT_PORT
    
    if auto_discover:
        discovered_host, discovered_port = discover_server_on_network(DEFAULT_PORT)
        if discovered_host:
            logger.info(f"4. Descubierto: {discovered_host}:{discovered_port}")
            save_config_file(discovered_host, discovered_port)
            return discovered_host, discovered_port
    
    logger.info(f"5. Default: {DEFAULT_HOST}:{DEFAULT_PORT}")
    return DEFAULT_HOST, DEFAULT_PORT


# ==================== DETECCION DE HARDWARE ====================
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


def get_optimal_config(hardware_info):
    tier = hardware_info.get("tier", "low")
    if tier == "high":
        return {"batch_size": 64, "num_workers": 4, "prefetch": True}
    elif tier == "medium":
        return {"batch_size": 32, "num_workers": 2, "prefetch": True}
    else:
        return {"batch_size": 16, "num_workers": 0, "prefetch": False}


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
class ChunkedDataset:
    def __init__(self, dataset_name='cifar10', data_dir='./data', batch_size=32, chunk_size=10):
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.full_dataset = self._load_dataset()
        self.total_samples = len(self.full_dataset)
        self.total_batches = (self.total_samples + batch_size - 1) // batch_size
        self.total_chunks = (self.total_batches + chunk_size - 1) // chunk_size
        logger.info(f"Dataset: {dataset_name} | {self.total_samples} muestras | {self.total_chunks} chunks")
    
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
    
    def get_sample_indices_for_chunks(self, chunk_indices):
        indices = []
        for chunk_idx in sorted(chunk_indices):
            start_batch = chunk_idx * self.chunk_size
            end_batch = min(start_batch + self.chunk_size, self.total_batches)
            start_sample = start_batch * self.batch_size
            end_sample = min(end_batch * self.batch_size, self.total_samples)
            indices.extend(range(start_sample, end_sample))
        return indices
    
    def create_dataloader(self, chunk_indices, num_workers=0):
        sample_indices = self.get_sample_indices_for_chunks(chunk_indices)
        subset = Subset(self.full_dataset, sample_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=False)


# ==================== WORKER ====================
class AsyncDistWorker:
    def __init__(self, server_host='localhost', server_port=8765, num_classes=10, data_dir='./data'):
        self.server_host = server_host
        self.server_port = server_port
        self.num_classes = num_classes
        self.data_dir = data_dir
        
        self.worker_id = None
        self.total_workers = None
        self.model = None
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.hardware_info = detect_hardware()
        self.optimal_config = get_optimal_config(self.hardware_info)
        
        self.chunked_dataset = None
        self.max_epochs = None  # Se recibe del servidor
        
        logger.info("=" * 60)
        logger.info("WORKER INICIANDO")
        logger.info(f"   Hardware: {self.hardware_info['cpu']}")
        logger.info(f"   Tier: {self.hardware_info['tier']}")
        logger.info(f"   Cores: {self.hardware_info['cores_physical']}F/{self.hardware_info['cores_logical']}T")
        logger.info(f"   RAM: {self.hardware_info['ram_gb']} GB")
        logger.info("=" * 60)

    def create_model(self, state_dict=None):
        self.model = EfficientNetLite0(num_classes=self.num_classes).to(self.device)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        logger.info(f"Modelo creado en {self.device}")

    def compute_gradients(self, data, target):
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
        self.model.load_state_dict(state_dict)

    async def run(self):
        logger.info(f"Conectando a {self.server_host}:{self.server_port}...")
        
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.server_host, self.server_port),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout conectando a {self.server_host}:{self.server_port}")
            return
        except Exception as e:
            logger.error(f"Error de conexion: {e}")
            return
        
        try:
            # 1. Handshake
            await send_msg(writer, {'type': 'handshake', 'hardware': self.hardware_info})
            
            # 2. Recibir ID + max_epochs del servidor
            response = await recv_msg(reader)
            if response['type'] == 'assign_id':
                self.worker_id = response['worker_id']
                self.total_workers = response['total_workers']
                server_batch_size = response['batch_size']
                chunk_size = response['chunk_size']
                dataset_name = response['dataset']
                self.max_epochs = response.get('max_epochs', 10)
                
                logger.info("=" * 60)
                logger.info("ASIGNACION RECIBIDA DEL SERVIDOR")
                logger.info(f"   Worker ID: {self.worker_id} / {self.total_workers}")
                logger.info(f"   Dataset: {dataset_name}")
                logger.info(f"   Batch size: {server_batch_size}")
                logger.info(f"   Chunk size: {chunk_size}")
                logger.info(f"   EPOCAS TOTALES: {self.max_epochs}")
                logger.info("=" * 60)
            
            # 3. Cargar dataset
            self.chunked_dataset = ChunkedDataset(
                dataset_name=dataset_name,
                data_dir=self.data_dir,
                batch_size=server_batch_size,
                chunk_size=chunk_size
            )
            
            # 4. Esperar modelo inicial
            logger.info("Esperando modelo inicial...")
            msg = await recv_msg(reader)
            
            if msg['type'] == 'epoch_start':
                epoch = msg['epoch']
                state_dict = msg['state_dict']
                chunk_assignment = msg['chunk_assignment']
                if 'max_epochs' in msg:
                    self.max_epochs = msg['max_epochs']
                
                self.create_model(state_dict)
                
                my_chunks = chunk_assignment.get(self.worker_id, [])
                logger.info(f"Epoca {epoch}/{self.max_epochs} | Chunks: {my_chunks[:10]}...")
                
                train_loader = self.chunked_dataset.create_dataloader(
                    my_chunks,
                    num_workers=self.optimal_config['num_workers']
                )
            else:
                raise ValueError(f"Mensaje inesperado: {msg['type']}")
            
            # 5. Bucle de entrenamiento
            global_step = 0
            current_epoch = epoch
            
            while current_epoch < self.max_epochs:
                epoch_loss = 0.0
                num_batches = 0
                
                logger.info(f"Worker {self.worker_id} | Epoca {current_epoch}/{self.max_epochs}")
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    gradients, loss = self.compute_gradients(data, target)
                    epoch_loss += loss
                    num_batches += 1
                    global_step += 1
                    
                    await send_msg(writer, {
                        'type': 'gradients',
                        'gradients': gradients,
                        'step': global_step,
                        'loss': loss,
                        'worker_id': self.worker_id,
                        'epoch': current_epoch
                    })
                    
                    try:
                        msg = await asyncio.wait_for(recv_msg(reader), timeout=60.0)
                        
                        if msg['type'] == 'model_update':
                            self.update_model(msg['state_dict'])
                            
                        elif msg['type'] == 'epoch_start':
                            new_epoch = msg['epoch']
                            new_state_dict = msg['state_dict']
                            new_assignment = msg['chunk_assignment']
                            if 'max_epochs' in msg:
                                self.max_epochs = msg['max_epochs']
                            
                            self.update_model(new_state_dict)
                            current_epoch = new_epoch
                            
                            my_chunks = new_assignment.get(self.worker_id, [])
                            logger.info(f"NUEVA EPOCA {current_epoch}/{self.max_epochs}")
                            
                            train_loader = self.chunked_dataset.create_dataloader(
                                my_chunks,
                                num_workers=self.optimal_config['num_workers']
                            )
                            break
                            
                        elif msg['type'] == 'training_done':
                            logger.info("Servidor indica fin del entrenamiento!")
                            return
                            
                    except asyncio.TimeoutError:
                        logger.warning("Timeout esperando modelo. Continuando...")
                    
                    if batch_idx % 20 == 0:
                        logger.info(f"   Worker {self.worker_id} | Epoch {current_epoch}/{self.max_epochs} | "
                                    f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f}")
                
                # Epoca completada
                avg_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"Worker {self.worker_id} | Epoca {current_epoch}/{self.max_epochs} COMPLETADA | Loss: {avg_loss:.4f}")
                
                await send_msg(writer, {
                    'type': 'epoch_complete',
                    'epoch': current_epoch,
                    'worker_id': self.worker_id,
                    'avg_loss': avg_loss
                })
                
                current_epoch += 1
            
            await send_msg(writer, {'type': 'done'})
            logger.info(f"Worker {self.worker_id} finalizo {self.max_epochs} epocas")
            
        except Exception as e:
            logger.error(f"Error en Worker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            writer.close()
            await writer.wait_closed()


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker de entrenamiento distribuido - EfficientNetLite-0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS DE USO:
================

1. Especificar IP (se guarda automaticamente):
   python worker.py --server-host 192.168.1.10

2. Usar configuracion guardada (sin argumentos):
   python worker.py

3. Descubrir servidor automaticamente:
   python worker.py --auto-discover

NOTA: El numero de epocas lo define el SERVIDOR (--epochs).
        """
    )
    
    parser.add_argument('--server-host', default=DEFAULT_HOST, help='IP del servidor')
    parser.add_argument('--server-port', type=int, default=DEFAULT_PORT, help='Puerto')
    parser.add_argument('--auto-discover', action='store_true', help='Descubrir servidor en red')
    parser.add_argument('--num-classes', type=int, default=10, help='Clases')
    parser.add_argument('--data-dir', default='./data', help='Directorio de datos')
    
    args = parser.parse_args()
    
    host, port = resolve_server_config(
        args.server_host, 
        args.server_port,
        auto_discover=args.auto_discover
    )
    
    worker = AsyncDistWorker(
        server_host=host,
        server_port=port,
        num_classes=args.num_classes,
        data_dir=args.data_dir
    )
    
    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        logger.info("Worker detenido")