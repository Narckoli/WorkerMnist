# worker.py (o main.py del worker)
import asyncio
import sys
import os

# Asegúrate de tener los imports correctos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Communication import connect_to_server, send_json, recv_json
from Config import WorkerConfig, config  # Importar tanto la clase como la instancia
from Dataset import load_dataset_chunk
from Training import training_loop
from Model import init_local_weights

async def main():
    """Punto de entrada principal del worker."""
    print("\n" + "="*60)
    print("WORKER DE ENTRENAMIENTO DISTRIBUIDO")
    print("="*60)
    
    # Configuración inicial - USAR LA CLASE WorkerConfig CORRECTAMENTE
    temp_config = WorkerConfig.from_input()
    config.SERVER_IP = temp_config.SERVER_IP
    config.PORT = temp_config.PORT
    
    # Conectar al servidor
    reader, writer = await connect_to_server(config.SERVER_IP, config.PORT)
    if not reader or not writer:
        print("No se pudo establecer conexión con el servidor")
        return
    
    try:
        # 1. RECIBIR ID DEL WORKER
        data = await recv_json(reader)
        if not data or data.get("type") != "worker_id":
            print("Error: No se recibió ID del servidor")
            return
        
        config.worker_id = data["worker_id"]
        print(f"\n✓ ID asignado: Worker {config.worker_id}")
        
        # 2. RECIBIR INFORMACIÓN DEL DATASET
        data = await recv_json(reader)
        if not data or data.get("type") != "dataset_info":
            print("Error: No se recibió información del dataset")
            return
        
        config.dataset_name = data["dataset_name"]
        config.input_size = data["input_size"]
        config.hidden_size = data.get("hidden_size", 128)
        config.output_size = data.get("output_size", 10)
        
        print(f"\n📊 Información del dataset recibida:")
        print(f"   Nombre: {config.dataset_name}")
        print(f"   Input size: {config.input_size}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Output size: {config.output_size}")
        
        # 3. RECIBIR CHUNK DE DATOS
        data = await recv_json(reader)
        if not data or data.get("type") != "dataset_chunk":
            print("Error: No se recibió el chunk de datos")
            return
        
        indices = data["indices"]
        print(f"\n✓ Chunk recibido: {len(indices)} muestras")
        
        # Cargar el dataset localmente usando la información recibida
        config.X_chunk, config.y_chunk = load_dataset_chunk(
            config.dataset_name, 
            indices
        )
        
        print(f"✓ Datos cargados: X shape={config.X_chunk.shape}, y shape={config.y_chunk.shape}")
        
        # 4. INICIALIZAR PESOS LOCALES (estructura para gradientes)
        config.local_weights_template = init_local_weights(
            config.input_size, 
            config.hidden_size, 
            config.output_size
        )
        
        # 5. ENVIAR CONFIRMACIÓN DE READY
        await send_json(writer, {
            "type": "worker_ready",
            "worker_id": config.worker_id,
            "dataset_name": config.dataset_name,
            "samples": len(config.X_chunk)
        })
        print(f"\n✓ Worker listo para entrenar")
        
        # 6. INICIAR BUCLE DE ENTRENAMIENTO
        config.print_info()
        await training_loop(reader, writer, config)
        
    except Exception as e:
        print(f"\n❌ Error en worker: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if writer:
            writer.close()
            await writer.wait_closed()
        print("\n🔌 Worker finalizado")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Worker interrumpido por el usuario")