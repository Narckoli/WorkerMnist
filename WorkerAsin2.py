"""
Worker 100% ASINCRONO - CORREGIDO.
- Protocolo simplificado: envia gradiente -> recibe model_update (1 solo mensaje)
- SIN lectura de mensajes "pendientes" despues de recv (causaba unpickling stack underflow)
- Entrena continuamente sin esperar a nadie
- Envia gradientes cada N batches (gradient_frequency)
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
        except Exception:
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
        import math
        out_ch = int(math.ceil(32 * width_mult / 8) * 8)
        self.stem = nn.Sequential(nn.Conv2d(3, out_ch, 3, 2, 1, bias=False), nn.BatchNorm2d(out_ch), Swish())
        blocks, in_ch = [], out_ch
        for expand_ratio, out_ch_cfg, kernel_size, stride, se_ratio, num_repeat in self.CONFIG:
            out_ch = int(math.ceil(out_ch_cfg * width_mult / 8) * 8)
            num_repeat = int(math.ceil(num_repeat * depth_mult))
            for i in range(num_repeat):
                blocks.append(MBConvBlock(in_ch, out_ch, expand_ratio, kernel_size, stride if i==0 else 1, se_ratio, dropout_rate))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        head_ch = int(math.ceil(1280 * width_mult / 8) * 8)
        self.head = nn.Sequential(nn.Conv2d(in_ch, head_ch, 1, bias=False), nn.BatchNorm2d(head_ch), Swish(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(head_ch, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ==================== PROTOCOLO ====================
async def send_msg(writer, data):
    """
    Envia un mensaje con length-prefix de 4 bytes (big-endian).
    Usa HIGHEST_PROTOCOL para consistencia con el servidor.
    """
    try:
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        length = struct.pack('>I', len(payload))
        writer.write(length + payload)
        await writer.drain()
        return True
    except Exception as e:
        logger.warning(f"Error enviando mensaje: {e}")
        return False

async def recv_msg(reader, timeout=30.0):
    """
    Recibe un mensaje con length-prefix de 4 bytes (big-endian).
    Lee EXACTAMENTE los bytes indicados - sin sobrelecura del buffer.
    """
    length_data = await asyncio.wait_for(reader.readexactly(4), timeout=timeout)
    length = struct.unpack('>I', length_data)[0]
    if length == 0:
        raise ValueError("Longitud de mensaje es 0")
    if length > 500_000_000:
        raise ValueError(f"Mensaje demasiado grande: {length} bytes")
    payload = await asyncio.wait_for(reader.readexactly(length), timeout=timeout)
    return pickle.loads(payload)


# ==================== DATASET ====================
class AsyncDataset:
    """
    Dataset para entrenamiento asincrono.
    - Divide el dataset en chunks
    - Cada worker recibe asignacion de chunks por round-robin
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
        self.assigned_chunks = [
            chunk_idx for chunk_idx in range(self.total_chunks)
            if chunk_idx % self.total_workers == self.worker_id
        ]

    def shuffle_chunks(self, seed=None):
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
    Worker completamente asincrono.

    Protocolo de comunicacion (estrictamente 1:1 con el servidor):
      Worker -> Servidor : {'type': 'gradient', ...}
      Servidor -> Worker : {'type': 'model_update', ...}   <- UN SOLO mensaje de respuesta

    NO hay mensajes 'gradient_received' intermedios.
    NO hay lecturas de mensajes "pendientes" fuera del ciclo principal.
    """

    def __init__(self, server_host='localhost', server_port=5000, num_classes=10,
                 data_dir='./data', gradient_frequency=2):
        self.server_host = server_host
        self.server_port = server_port
        self.num_classes = num_classes
        self.data_dir = data_dir
        self.gradient_frequency = gradient_frequency

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

        # Acumuladores de gradientes
        self.accumulated_gradients = {}
        self.accumulated_loss = 0.0
        self.accumulated_count = 0

        # Estadisticas
        self.total_batches_processed = 0
        self.total_gradients_sent = 0
        self.staleness_history = []

        logger.info("=" * 60)
        logger.info("WORKER 100% ASINCRONO - CORREGIDO")
        logger.info(f"   Hardware: {self.hardware_info['cpu']}")
        logger.info(f"   Tier: {self.hardware_info['tier']}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Gradient frequency: cada {gradient_frequency} batches")
        logger.info("=" * 60)

    def create_model(self, state_dict=None):
        self.model = EfficientNetLite0(num_classes=self.num_classes).to(self.device)
        if state_dict is not None:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in state_dict:
                        param.copy_(state_dict[name].to(self.device))
            logger.info(f"Modelo cargado ({len(state_dict)} tensores de parametros)")
        self.criterion = nn.CrossEntropyLoss()
        logger.info(f"Modelo listo en {self.device}")

    def zero_accumulated_gradients(self):
        self.accumulated_gradients = {}
        self.accumulated_loss = 0.0
        self.accumulated_count = 0

    def accumulate_batch_gradient(self, data, target):
        """
        Calcula gradientes para un batch y los acumula localmente.
        El servidor aplica optimizer.step() - aqui solo calculamos grads.
        """
        self.model.train()
        data, target = data.to(self.device), target.to(self.device)

        self.model.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_cpu = param.grad.detach().cpu().clone()
                if name not in self.accumulated_gradients:
                    self.accumulated_gradients[name] = grad_cpu
                else:
                    self.accumulated_gradients[name] += grad_cpu

        self.accumulated_loss += loss.item()
        self.accumulated_count += 1
        self.total_batches_processed += 1

        return loss.item()

    def get_averaged_gradients(self):
        if self.accumulated_count == 0:
            return {}
        return {
            name: grad / self.accumulated_count
            for name, grad in self.accumulated_gradients.items()
        }

    def apply_model_update(self, state_dict):
        """Carga los pesos recibidos del servidor en el modelo local."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state_dict:
                    param.copy_(state_dict[name].to(self.device))

    async def send_gradient_and_receive_model(self, writer, reader):
        """
        Envia gradientes acumulados al servidor y espera el model_update.

        Protocolo estricto:
          SEND: {'type': 'gradient', ...}
          RECV: {'type': 'model_update', ...}   <- exactamente 1 mensaje

        Retorna True para continuar, False para terminar.
        """
        if self.accumulated_count == 0:
            return True

        gradients = self.get_averaged_gradients()
        avg_loss = self.accumulated_loss / self.accumulated_count

        logger.info(f"Enviando gradiente | batches acumulados: {self.accumulated_count} | "
                   f"loss: {avg_loss:.4f} | version local: {self.local_model_version}")

        ok = await send_msg(writer, {
            'type': 'gradient',
            'gradients': gradients,
            'model_version': self.local_model_version,
            'loss': avg_loss,
            'worker_id': self.worker_id,
            'batch_count': self.accumulated_count
        })
        if not ok:
            logger.error("No se pudo enviar el gradiente al servidor")
            return False

        # Esperar la unica respuesta del servidor: model_update
        try:
            response = await recv_msg(reader, timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Timeout esperando model_update del servidor")
            # No limpiar gradientes - reintentar en el proximo ciclo
            return True
        except (asyncio.IncompleteReadError, ConnectionError) as e:
            logger.error(f"Conexion perdida esperando model_update: {e}")
            return False

        msg_type = response.get('type', '')

        if msg_type == 'model_update':
            new_version = response['model_version']
            staleness = new_version - self.local_model_version
            self.staleness_history.append(staleness)

            accepted = response.get('gradient_accepted', True)
            status = "aceptado" if accepted else "rechazado"
            logger.info(f"Modelo recibido: v{self.local_model_version} -> v{new_version} "
                       f"| staleness: {staleness} | gradiente: {status}")

            self.apply_model_update(response['state_dict'])
            self.local_model_version = new_version
            self.total_gradients_sent += 1
            self.zero_accumulated_gradients()

            if response.get('training_done', False):
                logger.info("Servidor indica: entrenamiento completado!")
                return False

            return True

        elif msg_type == 'training_done':
            logger.info("Servidor indica: training_done!")
            return False

        elif msg_type == 'heartbeat':
            # Heartbeat inesperado en este punto - ignorar y continuar
            logger.debug(f"Heartbeat recibido durante espera de model_update (version servidor: {response.get('model_version')})")
            self.zero_accumulated_gradients()
            return True

        else:
            logger.warning(f"Mensaje inesperado tipo '{msg_type}' - ignorando")
            self.zero_accumulated_gradients()
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
            # --- 1. Handshake ---
            ok = await send_msg(writer, {'type': 'handshake', 'hardware': self.hardware_info})
            if not ok:
                logger.error("Error enviando handshake")
                return

            # --- 2. Recibir configuracion ---
            response = await recv_msg(reader, timeout=30.0)
            if response['type'] != 'assign_id':
                raise ValueError(f"Se esperaba 'assign_id', recibido: {response['type']}")

            self.worker_id = response['worker_id']
            self.total_workers = response['total_workers']
            server_batch_size = response['batch_size']
            dataset_name = response['dataset']
            self.max_updates = response.get('max_updates', 1000)

            logger.info(f"Asignado: Worker {self.worker_id}/{self.total_workers}")
            logger.info(f"   Batch size: {server_batch_size} | Max updates: {self.max_updates}")

            # --- 3. Cargar dataset ---
            self.dataset = AsyncDataset(
                dataset_name=dataset_name,
                data_dir=self.data_dir,
                batch_size=server_batch_size,
                chunk_size=10,
                worker_id=self.worker_id,
                total_workers=self.total_workers
            )

            # --- 4. Recibir modelo inicial ---
            logger.info("Esperando modelo inicial del servidor...")
            msg = await recv_msg(reader, timeout=60.0)

            if msg['type'] != 'model_update':
                raise ValueError(f"Se esperaba 'model_update' inicial, recibido: {msg['type']}")

            self.create_model(msg['state_dict'])
            self.local_model_version = msg['model_version']
            logger.info(f"Modelo inicial recibido (version {self.local_model_version})")

            # --- 5. Bucle de entrenamiento asincrono ---
            logger.info("=" * 60)
            logger.info("INICIANDO ENTRENAMIENTO ASINCRONO CONTINUO")
            logger.info("=" * 60)

            batch_iterator = self.dataset.get_all_batches_iterator()
            self.zero_accumulated_gradients()
            running = True

            while running:
                # Acumular gradientes de N batches
                for _ in range(self.gradient_frequency):
                    try:
                        (data, target), epoch = next(batch_iterator)
                        loss = self.accumulate_batch_gradient(data, target)

                        if self.total_batches_processed % 20 == 0:
                            avg_loss = self.accumulated_loss / max(self.accumulated_count, 1)
                            logger.info(f"Batch {self.total_batches_processed} | "
                                       f"Loss: {loss:.4f} | Acum loss: {avg_loss:.4f} | "
                                       f"Version local: {self.local_model_version}")

                    except StopIteration:
                        # Generador infinito - no deberia ocurrir
                        break
                    except Exception as e:
                        logger.error(f"Error procesando batch: {e}")
                        continue

                # Enviar gradientes y recibir modelo actualizado
                if self.accumulated_count > 0:
                    running = await self.send_gradient_and_receive_model(writer, reader)

            # --- 6. Finalizar ---
            await send_msg(writer, {'type': 'done'})

            avg_staleness = (sum(self.staleness_history) / len(self.staleness_history)
                            if self.staleness_history else 0)
            logger.info("=" * 60)
            logger.info("WORKER FINALIZADO")
            logger.info(f"   Batches procesados:  {self.total_batches_processed}")
            logger.info(f"   Gradientes enviados: {self.total_gradients_sent}")
            logger.info(f"   Staleness promedio:  {avg_staleness:.2f}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error en Worker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                writer.close()
                await asyncio.wait_for(writer.wait_closed(), timeout=2.0)
            except Exception:
                pass


# ==================== MAIN ====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Worker 100% Asincrono - Corregido',
        epilog="Ejemplo: python WorkerAsin2.py --server-host 192.168.0.11"
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