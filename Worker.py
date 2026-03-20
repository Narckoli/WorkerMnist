# worker/worker.py
import asyncio
import sys
import os

# Asegurar imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Config import WorkerConfig
from Communication import send_json, recv_json, connect_to_server
from Dataset import load_mnist_dataset, extract_chunk
from Training import training_loop

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
        # Procesar mensajes iniciales
        X_chunk = None
        y_chunk = None
        
        print("\n⏳ Esperando asignación de datos del servidor...")
        
        while X_chunk is None:
            data = await recv_json(reader)
            if data is None:
                print("❌ Conexión cerrada por el servidor")
                return
            
            msg_type = data.get("type")
            
            if msg_type == "worker_id":
                config.worker_id = data['worker_id']
                print(f"✓ Mi ID asignado: Worker {config.worker_id}")
            
            elif msg_type == "dataset_chunk":
                indices = data["indices"]
                X_chunk, y_chunk = extract_chunk(X_train, y_train, indices)
                print(f"✓ Chunk recibido: {len(indices)} muestras")
        
        # ===== IMPORTANTE: Enviar señal de READY =====
        print("\n🚦 Worker listo para entrenar. Enviando señal READY al servidor...")
        await send_json(writer, {"type": "worker_ready"})
        print("✓ Señal READY enviada")
        
        # Iniciar bucle de entrenamiento
        print("\n🎯 Esperando inicio del entrenamiento...")
        await training_loop(reader, writer, X_chunk, y_chunk, config)
        
    except KeyboardInterrupt:
        print("\n\n⛔ Worker interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        writer.close()
        await writer.wait_closed()
        print("🔌 Conexión cerrada")

if __name__ == "__main__":
    try:
        asyncio.run(start_worker())
    except KeyboardInterrupt:
        print("\n\n👋 Worker finalizado")