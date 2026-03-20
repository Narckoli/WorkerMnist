# worker/training.py
import time
import numpy as np

from Communication import send_json, recv_json
from Model import train_epoch
from Metrics import metrics

async def training_loop(reader, writer, X_chunk, y_chunk, config):
    """Bucle principal de entrenamiento del worker."""
    
    try:
        while True:
            # Recibir mensaje del servidor
            data = await recv_json(reader)
            
            if data is None:
                print("Conexión cerrada por el servidor")
                break

            msg_type = data.get("type")
            
            if msg_type == "weights":
                epoch = data.get("epoch", "?")
                print(f"\n>>> ÉPOCA {epoch} <<<")
                
                # Reconstruir pesos
                weights = {
                    "W1": np.array(data["W1"]),
                    "b1": np.array(data["b1"]),
                    "W2": np.array(data["W2"]),
                    "b2": np.array(data["b2"])
                }
                
                # Medir tiempo de entrenamiento
                start_time = time.time()
                
                # Entrenar época
                grads, loss = train_epoch(X_chunk, y_chunk, weights)
                
                epoch_time = time.time() - start_time
                
                print(f"✓ Loss calculada: {loss:.6f}")
                print(f"✓ Tiempo: {epoch_time:.2f}s")
                
                # Guardar métricas locales
                metrics.add_epoch_result(loss, epoch_time)
                
                # Enviar gradientes al servidor
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
                
                # Mostrar resumen local
                metrics.print_summary()
                break

            else:
                print(f"⚠ Tipo de mensaje desconocido: {msg_type}")
                
    except Exception as e:
        print(f"Error en training_loop: {e}")
        import traceback
        traceback.print_exc()