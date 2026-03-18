# worker.py
import asyncio
import time
import numpy as np

from Config import config, WorkerConfig
from Communication import send_json, recv_json, connect_to_server
from Dataset import load_mnist_dataset, extract_chunk
from Training import train_epoch, verify_weights
from Metrics import metrics

# ======================
# Manejador de mensajes
# ======================
async def handle_messages(reader: asyncio.StreamReader, 
                          writer: asyncio.StreamWriter,
                          X_train: np.ndarray,
                          y_train: np.ndarray):
    """Maneja los mensajes entrantes del servidor."""
    
    X_chunk = None
    y_chunk = None
    
    while True:
        print("\n" + "-" * 40)
        data = await recv_json(reader)
        
        if data is None:
            print("Conexión cerrada por el servidor")
            break

        msg_type = data.get("type")
        print(f"Mensaje recibido: {msg_type}")

        if msg_type == "worker_id":
            config.worker_id = data['worker_id']
            print(f"✓ Mi ID asignado: {config.worker_id}")

        elif msg_type == "dataset_chunk":
            indices = data["indices"]
            X_chunk, y_chunk = extract_chunk(X_train, y_train, indices)
            print(f"✓ Chunk recibido: {len(indices)} muestras")

        elif msg_type == "weights":
            epoch = data.get("epoch", "?")
            print(f"\n>>> ÉPOCA {epoch} <<<")
            
            # Reconstruir pesos
            weights = {
                "W1": np.array(data["W1"]),
                "b1": np.array(data["b1"]),
                "W2": np.array(data["W2"]),
                "b2": np.array(data["b2"])
            }
            
            # Verificar pesos (debug)
            verify_weights(weights)
            
            # Medir tiempo de entrenamiento
            start_time = time.time()
            
            # Entrenar época
            grads, loss = train_epoch(X_chunk, y_chunk, weights)
            
            epoch_time = time.time() - start_time
            
            print(f"✓ Loss calculada: {loss:.4f}")
            print(f"✓ Tiempo de entrenamiento: {epoch_time:.2f}s")
            
            # Guardar métricas
            metrics.add_epoch_result(loss, epoch_time)
            
            # Enviar resultados al servidor
            await send_json(writer, {
                "type": "gradients",
                "grads": {
                    "W1": grads["W1"].tolist(),
                    "b1": grads["b1"].tolist(),
                    "W2": grads["W2"].tolist(),
                    "b2": grads["b2"].tolist()
                },
                "loss": float(loss)
            })
            
            print("✓ Gradientes enviados al servidor")

        elif msg_type == "training_complete":
            print(f"\n{'='*50}")
            print(data.get("message", "Entrenamiento finalizado"))
            print(f"{'='*50}")
            break

        else:
            print(f"⚠ Tipo de mensaje desconocido: {msg_type}")

# ======================
# Worker Principal
# ======================
async def start_worker():
    """Punto de entrada principal del worker."""
    print("=" * 50)
    print("WORKER DE ENTRENAMIENTO DISTRIBUIDO MNIST (ASYNC)")
    print("=" * 50)
    
    # Configuración
    config = WorkerConfig.from_input()
    
    # Cargar dataset
    print("\nCargando dataset...")
    X_train, y_train = load_mnist_dataset()

    # Conectar al servidor
    reader, writer = await connect_to_server(config.SERVER_IP, config.PORT)
    if not reader or not writer:
        return

    try:
        # Manejar mensajes
        await handle_messages(reader, writer, X_train, y_train)
        
        # Mostrar resumen local
        metrics.print_summary()
        
    except KeyboardInterrupt:
        print("\n\nInterrumpido por el usuario")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()
        print("Conexión cerrada")

if __name__ == "__main__":
    try:
        asyncio.run(start_worker())
    except KeyboardInterrupt:
        print("\n\nWorker interrumpido por el usuario")