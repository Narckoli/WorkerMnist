# worker.py - Versión corregida con prefijo de longitud
import asyncio
import json
import struct
import numpy as np
import sys
import os

# Agregar directorio padre al path si es necesario
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Intentar importar configuración, si no existe, usar valores por defecto
from Config import WorkerConfig
SERVER_IP = WorkerConfig.SERVER_IP
PORT = WorkerConfig.PORT
USE_CNN = WorkerConfig.USE_CNN


class FederatedWorker:
    """
    Worker unificado con protocolo de comunicación correcto
    """
    def __init__(self, server_host=None, server_port=None):
        # Configuración
        self.server_host = server_host if server_host else SERVER_IP
        self.server_port = server_port if server_port else PORT
        self.use_cnn = USE_CNN
        
        # Estado
        self.reader = None
        self.writer = None
        self.worker_id = None
        self.dataset_name = None
        self.input_size = None
        self.X_chunk = None
        self.y_chunk = None
        
        print(f"📡 Configuración:")
        print(f"   Servidor: {self.server_host}:{self.server_port}")
        print(f"   Modelo: {'CNN' if self.use_cnn else 'MLP'}")
    
    async def connect(self):
        """Conecta al servidor"""
        print(f"🔌 Conectando a {self.server_host}:{self.server_port}...")
        self.reader, self.writer = await asyncio.open_connection(
            self.server_host, self.server_port
        )
        print(f"✅ Conectado al servidor")
    
    async def send_json(self, data: dict):
        """
        Envía un mensaje JSON con prefijo de longitud.
        IMPORTANTE: Debe coincidir con el servidor.
        """
        try:
            message = json.dumps(data).encode('utf-8')
            length = struct.pack(">I", len(message))
            self.writer.write(length + message)
            await self.writer.drain()
        except Exception as e:
            print(f"[ERROR] send_json: {e}")
            raise
    
    async def recv_json(self):
        """
        Recibe un mensaje JSON con prefijo de longitud.
        IMPORTANTE: Debe coincidir con el servidor.
        """
        try:
            # Leer los primeros 4 bytes (longitud del mensaje)
            raw_length = await self.reader.read(4)
            if not raw_length or len(raw_length) < 4:
                print(f"[DEBUG] No se recibió la longitud del mensaje")
                return None
            
            # Desempaquetar longitud (big-endian)
            message_length = struct.unpack(">I", raw_length)[0]
            
            if message_length <= 0 or message_length > 10 * 1024 * 1024:  # Máximo 10MB
                print(f"[ERROR] Longitud inválida: {message_length}")
                return None
            
            # Leer el mensaje completo
            data = b""
            while len(data) < message_length:
                chunk_size = min(8192, message_length - len(data))
                packet = await self.reader.read(chunk_size)
                if not packet:
                    print(f"[DEBUG] Conexión cerrada durante lectura del mensaje")
                    return None
                data += packet
            
            # Decodificar JSON
            return json.loads(data.decode('utf-8'))
            
        except struct.error as e:
            print(f"[ERROR] Error desempaquetando longitud: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error decodificando JSON: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] recv_json: {e}")
            return None
    
    async def setup(self):
        """Configuración inicial del worker"""
        print(f"[Worker] Esperando configuración del servidor...")
        
        # 1. Recibir ID del worker
        msg = await self.recv_json()
        if not msg:
            print(f"[ERROR] No se recibió el ID del worker")
            return False
        
        if msg.get("type") == "worker_id":
            self.worker_id = msg["worker_id"]
            print(f"[Worker {self.worker_id}] ✓ ID recibido")
        else:
            print(f"[ERROR] Mensaje inesperado: {msg}")
            return False
        
        # 2. Recibir información del dataset
        msg = await self.recv_json()
        if not msg:
            print(f"[ERROR] No se recibió información del dataset")
            return False
        
        if msg.get("type") == "dataset_info":
            self.dataset_name = msg.get("dataset_name")
            self.input_size = msg.get("input_size")
            print(f"[Worker {self.worker_id}] ✓ Dataset: {self.dataset_name}, Input size: {self.input_size}")
        else:
            print(f"[ERROR] Mensaje inesperado: {msg}")
            return False
        
        # 3. Recibir chunk de datos
        msg = await self.recv_json()
        if not msg:
            print(f"[ERROR] No se recibió el chunk de datos")
            return False
        
        if msg.get("type") == "dataset_chunk":
            indices = msg.get("indices")
            print(f"[Worker {self.worker_id}] ✓ Chunk recibido: {len(indices)} muestras")
            self.indices = np.array(indices)
        else:
            print(f"[ERROR] Mensaje inesperado: {msg}")
            return False
        
        # 4. Cargar datos reales
        await self.load_data()
        
        # 5. Enviar confirmación de que está listo
        await self.send_json({"type": "worker_ready", "worker_id": self.worker_id})
        print(f"[Worker {self.worker_id}] ✅ Worker listo para entrenar")
        
        return True
    
    async def load_data(self):
        """Carga los datos reales usando los índices recibidos"""
        try:
            from Dataset import load_dataset_by_name
        except ImportError:
            print(f"[ERROR] No se puede importar Dataset.py")
            return
        
        print(f"[Worker {self.worker_id}] Cargando dataset {self.dataset_name}...")
        X_train, y_train, _, _ = load_dataset_by_name(self.dataset_name)
        
        self.X_chunk = X_train[self.indices]
        self.y_chunk = y_train[self.indices]
        
        print(f"[Worker {self.worker_id}] Datos cargados: {len(self.X_chunk)} muestras")
        print(f"   X shape: {self.X_chunk.shape}")
        print(f"   y shape: {self.y_chunk.shape}")
    
    async def train_loop(self):
        """Bucle principal de entrenamiento"""
        from Model import compute_gradients
        
        print(f"[Worker {self.worker_id}] Iniciando bucle de entrenamiento...")
        
        epoch = 0
        while True:
            try:
                # Recibir mensaje del servidor
                msg = await self.recv_json()
                
                if msg is None:
                    print(f"[Worker {self.worker_id}] Conexión cerrada por el servidor")
                    break
                
                msg_type = msg.get("type")
                
                if msg_type == "weights":
                    # Extraer pesos
                    weights = {
                        "W1": np.array(msg["W1"]),
                        "b1": np.array(msg["b1"]),
                        "W2": np.array(msg["W2"]),
                        "b2": np.array(msg["b2"])
                    }
                    epoch = msg.get("epoch", 0)
                    
                    print(f"[Worker {self.worker_id}] Época {epoch} - Calculando gradientes...")
                    
                    # Calcular gradientes
                    gradients, loss = compute_gradients(self.X_chunk, self.y_chunk, weights)
                    
                    # Enviar gradientes al servidor
                    await self.send_json({
                        "type": "gradients",
                        "worker_id": self.worker_id,
                        "grads": {
                            "W1": gradients["W1"].tolist(),
                            "b1": gradients["b1"].tolist(),
                            "W2": gradients["W2"].tolist(),
                            "b2": gradients["b2"].tolist()
                        },
                        "loss": loss,
                        "epoch": epoch
                    })
                    
                    print(f"[Worker {self.worker_id}] Gradientes enviados (loss: {loss:.4f})")
                
                elif msg_type == "stop":
                    print(f"[Worker {self.worker_id}] Señal de parada recibida")
                    break
                
                else:
                    print(f"[Worker {self.worker_id}] Mensaje desconocido: {msg_type}")
                    
            except asyncio.TimeoutError:
                print(f"[Worker {self.worker_id}] Timeout, continuando...")
                continue
            except Exception as e:
                print(f"[Worker {self.worker_id}] Error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print(f"[Worker {self.worker_id}] Entrenamiento completado")
    
    async def run(self):
        """Ejecuta el worker"""
        try:
            await self.connect()
            
            if await self.setup():
                await self.train_loop()
            else:
                print(f"[Worker] ❌ Error en configuración inicial")
                
        except ConnectionRefusedError:
            print(f"❌ No se pudo conectar al servidor {self.server_host}:{self.server_port}")
            print("   Verifica que:")
            print(f"   1. El servidor esté ejecutándose")
            print(f"   2. La IP {self.server_host} sea correcta")
            print(f"   3. No haya firewall bloqueando el puerto {self.server_port}")
        except Exception as e:
            print(f"❌ Error en worker: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()
                print(f"[Worker] Conexión cerrada")

async def main():
    """Punto de entrada principal"""
    print("="*50)
    print("Iniciando Worker Federado")
    print("="*50)
    from Config import SERVER_IP, PORT
    # Permitir pasar IP y puerto como argumentos
    server_host = SERVER_IP
    server_port = PORT
    
    if len(sys.argv) > 1:
        server_host = sys.argv[1]
    if len(sys.argv) > 2:
        server_port = int(sys.argv[2])
    
    worker = FederatedWorker(server_host, server_port)
    await worker.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Worker interrumpido por el usuario")