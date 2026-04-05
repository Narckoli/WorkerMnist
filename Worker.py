# Worker.py
import asyncio
import json
import sys
import numpy as np

# ── Configuración ─────────────────────────────────────────────────────────────
HOST = '192.168.0.3'
PORT = 5000
BATCH_SIZE = 64   # baja a 64 si te quedas sin RAM
# ─────────────────────────────────────────────────────────────────────────────

_BUFFER_LIMIT = 100 * 1024 * 1024


async def send_json(writer, data: dict):
    line = json.dumps(data) + '\n'
    writer.write(line.encode())
    await writer.drain()


async def recv_json(reader) -> dict | None:
    try:
        line = await reader.readline()
        if not line:
            return None
        return json.loads(line.decode())
    except (json.JSONDecodeError, asyncio.IncompleteReadError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Carga de datos
# ══════════════════════════════════════════════════════════════════════════════

def load_chunk(dataset_name: str, indices: list):
    from torchvision import datasets, transforms

    idx = np.array(indices, dtype=np.int64)

    if dataset_name == 'mnist':
        ds = datasets.MNIST(root='./data', train=True, download=True,
                            transform=transforms.ToTensor())
        X = ds.data.numpy()[idx].reshape(-1, 784) / 255.0
        y = ds.targets.numpy()[idx]

    elif dataset_name in ('cifar10', 'cifar-10'):
        ds = datasets.CIFAR10(root='./data', train=True,
                              download=True, transform=None)
        X = ds.data[idx].reshape(-1, 3072).astype(np.float32) / 255.0
        y = np.array(ds.targets)[idx]

    else:
        raise ValueError(f"Dataset no soportado: {dataset_name}")

    print(f"   Chunk cargado — X:{X.shape}, y:{y.shape}")
    return X.astype(np.float32), y.astype(np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# Gradientes MLP (numpy puro)
# ══════════════════════════════════════════════════════════════════════════════

def mlp_compute_gradients(X, y, weights):
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    m = X.shape[0]

    Z1 = X @ W1 + b1
    A1 = np.maximum(0, Z1)
    Z2 = A1 @ W2 + b2
    expZ = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
    A2 = expZ / expZ.sum(axis=1, keepdims=True)

    loss = -np.log(A2[np.arange(m), y] + 1e-9).mean()

    dZ2 = A2.copy()
    dZ2[np.arange(m), y] -= 1
    dZ2 /= m
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(axis=0)
    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, float(loss)


# ══════════════════════════════════════════════════════════════════════════════
# CNN (PyTorch) — gradientes + buffers BatchNorm
# ══════════════════════════════════════════════════════════════════════════════

def _build_cnn():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class CIFAR10CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn3   = nn.BatchNorm2d(128)
            self.pool  = nn.MaxPool2d(2, 2)
            self.fc1   = nn.Linear(128 * 4 * 4, 256)
            self.fc2   = nn.Linear(256, 10)
            self.drop  = nn.Dropout(0.5)

        def forward(self, x):
            x = self.pool(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool(torch.relu(self.bn2(self.conv2(x))))
            x = self.pool(torch.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 128 * 4 * 4)
            x = torch.relu(self.fc1(x))
            x = self.drop(x)
            return self.fc2(x)

    return CIFAR10CNN().to(device), device


def cnn_compute_gradients(X_np, y_np, weights_dict, model, device,
                          batch_size=BATCH_SIZE):
    """
    1. Carga state_dict completo (respetando dtypes).
    2. Acumula gradientes sobre mini-batches.
    3. Retorna:
         - gradients: solo parámetros entrenables (para el servidor)
         - bn_buffers: running_mean, running_var, num_batches_tracked
                       (para que el servidor actualice sus propios buffers)
         - loss
    """
    import torch
    import torch.nn.functional as F

    # Cargar state_dict completo
    current_sd = model.state_dict()
    new_sd = {}
    for k, v in weights_dict.items():
        arr = np.array(v)
        original_dtype = current_sd[k].dtype
        new_sd[k] = torch.tensor(arr, dtype=original_dtype).to(device)
    model.load_state_dict(new_sd, strict=True)
    model.train()
    model.zero_grad()

    n          = len(X_np)
    total_loss = 0.0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_b = torch.tensor(
            X_np[start:end].reshape(-1, 3, 32, 32),
            dtype=torch.float32).to(device)
        y_b = torch.tensor(y_np[start:end], dtype=torch.long).to(device)

        out  = model(X_b)
        loss = F.cross_entropy(out, y_b) * (end - start) / n
        loss.backward()
        total_loss += loss.item()

    # Gradientes de parámetros entrenables
    grads = {
        name: param.grad.detach().cpu().numpy()
        for name, param in model.named_parameters()
        if param.grad is not None
    }

    # Buffers de BatchNorm actualizados por el forward pass
    param_keys = {name for name, _ in model.named_parameters()}
    bn_buffers = {}
    for k, v in model.state_dict().items():
        if k not in param_keys:   # es un buffer, no un parámetro
            val = v.cpu().numpy()
            bn_buffers[k] = val.tolist() if isinstance(val, np.ndarray) else int(val)

    n_batches = (n + batch_size - 1) // batch_size
    print(f"   {n_batches} mini-batches procesados (n={n})")

    return grads, bn_buffers, float(total_loss)


# ══════════════════════════════════════════════════════════════════════════════
# Worker principal
# ══════════════════════════════════════════════════════════════════════════════

class Worker:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.reader = self.writer = None
        self.worker_id = self.dataset_name = None
        self.input_size = self.model_type = None
        self.X_chunk = self.y_chunk = None
        self._cnn_model = self._cnn_device = None

    async def connect(self):
        print(f"Conectando a {self.host}:{self.port}...")
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port, limit=_BUFFER_LIMIT)
        print("✓ Conectado al servidor")

    async def setup(self) -> bool:
        print("Esperando configuración del servidor...")

        msg = await recv_json(self.reader)
        if not msg or msg.get("type") != "id":
            print(f"✗ Paso 1 inesperado: {msg}")
            return False
        self.worker_id = msg["worker_id"]
        print(f"[Worker {self.worker_id}] ✓ ID recibido")

        msg = await recv_json(self.reader)
        if not msg or msg.get("type") != "dataset_info":
            print(f"✗ Paso 2 inesperado: {msg}")
            return False
        self.dataset_name = msg["dataset"]
        self.input_size   = msg["input_size"]
        self.model_type   = msg.get("model_type", "mlp")
        print(f"[Worker {self.worker_id}] ✓ Dataset:{self.dataset_name} "
              f"input:{self.input_size} modelo:{self.model_type}")

        msg = await recv_json(self.reader)
        if not msg or msg.get("type") != "chunk":
            print(f"✗ Paso 3 inesperado: {msg}")
            return False
        indices = msg["indices"]
        print(f"[Worker {self.worker_id}] ✓ Chunk recibido ({len(indices)} muestras)")

        print(f"[Worker {self.worker_id}] Cargando datos...")
        self.X_chunk, self.y_chunk = load_chunk(self.dataset_name, indices)

        if self.model_type == 'cnn':
            print(f"[Worker {self.worker_id}] Inicializando CNN...")
            self._cnn_model, self._cnn_device = _build_cnn()
            print(f"[Worker {self.worker_id}] ✓ CNN en {self._cnn_device}")

        await send_json(self.writer, {"type": "ready"})
        print(f"[Worker {self.worker_id}] ✓ Listo para entrenar")
        return True

    async def train_loop(self):
        print(f"[Worker {self.worker_id}] Iniciando bucle de entrenamiento...")

        while True:
            msg = await recv_json(self.reader)
            if msg is None:
                print(f"[Worker {self.worker_id}] Servidor cerró la conexión")
                break

            mtype = msg.get("type")

            if mtype == "weights":
                epoch        = msg.get("epoch", "?")
                weights_data = msg["weights"]

                print(f"[Worker {self.worker_id}] Época {epoch} "
                      "— calculando gradientes...")

                if self.model_type == 'cnn':
                    grads, bn_buffers, loss = cnn_compute_gradients(
                        self.X_chunk, self.y_chunk,
                        weights_data, self._cnn_model, self._cnn_device)

                    # Incluir buffers BatchNorm en el mensaje de gradientes
                    grads_serial = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in grads.items()
                    }
                    grads_serial['__bn_buffers__'] = bn_buffers

                else:
                    weights = {k: np.array(v) for k, v in weights_data.items()}
                    grads, loss = mlp_compute_gradients(
                        self.X_chunk, self.y_chunk, weights)
                    grads_serial = {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in grads.items()
                    }

                await send_json(self.writer, {
                    "type":      "gradients",
                    "worker_id": self.worker_id,
                    "gradients": grads_serial,
                    "loss":      loss,
                    "epoch":     epoch,
                })
                print(f"[Worker {self.worker_id}] ✓ Gradientes enviados "
                      f"(loss:{loss:.4f})")

            elif mtype == "ping":
                await send_json(self.writer, {"type": "pong"})

            elif mtype == "shutdown":
                print(f"[Worker {self.worker_id}] Apagado recibido")
                break

            else:
                print(f"[Worker {self.worker_id}] Mensaje desconocido: {mtype}")

    async def run(self):
        try:
            await self.connect()
            if await self.setup():
                await self.train_loop()
            else:
                print("✗ Error en setup")
        except ConnectionRefusedError:
            print(f"✗ No se pudo conectar a {self.host}:{self.port}")
        except Exception as e:
            print(f"✗ Error en worker: {e}")
            import traceback; traceback.print_exc()
        finally:
            if self.writer:
                try:
                    self.writer.close()
                    await self.writer.wait_closed()
                except Exception:
                    pass
            print(f"[Worker {self.worker_id}] Conexión cerrada")


async def main():
    host = sys.argv[1] if len(sys.argv) > 1 else HOST
    port = int(sys.argv[2]) if len(sys.argv) > 2 else PORT
    print("=" * 50)
    print(f"Worker — Servidor: {host}:{port}")
    print("=" * 50)
    await Worker(host, port).run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠ Worker detenido por el usuario")