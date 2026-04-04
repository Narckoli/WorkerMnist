# Training.py - Servidor
"""
Correcciones respecto a la versión anterior:

1. SINCRONIZACIÓN DE PESOS COMPLETA
   named_parameters() omite los buffers de BatchNorm (running_mean,
   running_var, num_batches_tracked). Ahora se usa state_dict() completo
   tanto para enviar pesos al worker como para cargarlos en el modelo
   del servidor antes de evaluar.

2. MOMENTUM EN LA ACTUALIZACIÓN DE PESOS (SGD con momentum=0.9)
   La actualización simple  w ← w - lr·g  hace que el loss oscile sin
   converger en redes profundas. Se añade un acumulador de velocidad
   que suaviza la trayectoria de descenso.

3. LEARNING RATE AUTOMÁTICO PARA CNN
   Si el usuario pone lr >= 0.01 con CNN, se ajusta automáticamente a
   0.001 para evitar divergencia.
"""

import asyncio
import time
import numpy as np

from Config import state
from Communication import send_json
from Export_Metrics import save_epoch_metrics, save_final_summary


# ── Helpers internos ──────────────────────────────────────────────────────────

def _evaluate(X, y):
    """Evalúa el modelo global actual. Carga el state_dict completo en CNN."""
    if state.model_type == 'cnn' and state.model:
        import torch
        full_sd = {
            k: torch.tensor(np.array(v), dtype=torch.float32)
            for k, v in state.global_weights.items()
        }
        state.model.model.load_state_dict(full_sd, strict=True)
        return state.model.evaluate(state.X_test, state.y_test)
    else:
        from Model import evaluate_model
        return evaluate_model(state.X_test, state.y_test, state.global_weights)


def _average_gradients(gradients_list):
    if not gradients_list:
        return {}
    avg = {}
    for key in gradients_list[0]:
        avg[key] = np.mean([np.array(g[key]) for g in gradients_list], axis=0)
    return avg


def _apply_gradients_with_momentum(weights, grads, velocity,
                                   lr, momentum=0.9):
    """
    SGD con momentum:
        v ← momentum·v + lr·g
        w ← w - v
    Solo actualiza las claves que son parámetros entrenables (están en grads).
    Las claves de BatchNorm que no tienen gradiente (running_mean, etc.)
    se copian sin cambio.
    """
    new_weights  = {}
    new_velocity = {}
    for key in weights:
        if key not in grads:
            # Buffer de BatchNorm u otro tensor sin gradiente — copiar tal cual
            new_weights[key]  = weights[key]
            new_velocity[key] = velocity.get(key, np.zeros_like(weights[key]))
        else:
            g = np.clip(np.array(grads[key]), -5.0, 5.0)
            v = momentum * velocity.get(key, np.zeros_like(g)) + lr * g
            new_weights[key]  = weights[key] - v
            new_velocity[key] = v
    return new_weights, new_velocity


async def send_weights_to_worker(writer, weights: dict, epoch: int):
    """Envía el state_dict completo (incluye buffers BatchNorm)."""
    serializable = {
        k: (v.tolist() if isinstance(v, np.ndarray) else
            v.cpu().numpy().tolist() if hasattr(v, 'cpu') else v)
        for k, v in weights.items()
    }
    await send_json(writer, {
        "type":       "weights",
        "epoch":      epoch,
        "model_type": state.model_type,
        "weights":    serializable,
    })


# ── Bucle principal ───────────────────────────────────────────────────────────

async def training_loop():
    print("\n" + "=" * 70)
    print("INICIO DE ENTRENAMIENTO")
    print("=" * 70)

    # Ajustar lr para CNN si el usuario puso un valor demasiado alto
    if state.model_type == 'cnn' and state.learning_rate >= 0.01:
        old_lr = state.learning_rate
        state.learning_rate = 0.001
        print(f"⚠  lr={old_lr} demasiado alto para CNN — ajustado a 0.001")

    async with state.lock:
        state.worker_gradients    = {}
        state.worker_losses       = {}
        state.all_workers_ready   = asyncio.Event()
        state.current_epoch       = 0
        state.training_start_time = time.time()
        state.epoch_events        = {}

    # Para CNN: usar state_dict completo como pesos globales
    if state.model_type == 'cnn' and state.model:
        state.global_weights = {
            k: v.cpu().numpy()
            for k, v in state.model.model.state_dict().items()
        }
        print(f"✓ Pesos CNN desde state_dict completo "
              f"({len(state.global_weights)} tensores)")

    # Inicializar acumulador de momentum
    velocity = {k: np.zeros_like(v) for k, v in state.global_weights.items()}

    # Esperar workers
    print("Esperando que todos los workers completen su setup...")
    while True:
        async with state.lock:
            ready = state.check_all_workers_ready_for_training()
        if ready:
            break
        await asyncio.sleep(0.5)
    print("✓ Todos los workers listos\n")

    # Evaluación inicial
    init_loss, init_acc = _evaluate(state.X_test, state.y_test)
    print(f"Evaluación inicial — Loss:{init_loss:.4f}  Accuracy:{init_acc:.4f}\n")

    # ── Épocas ────────────────────────────────────────────────────────────────
    for epoch in range(state.max_epochs):
        async with state.lock:
            state.current_epoch    = epoch
            state.epoch_start_time = time.time()
            state.worker_gradients.clear()
            state.worker_losses.clear()
            state.all_workers_ready.clear()

        print("=" * 70)
        print(f"ÉPOCA {epoch + 1}/{state.max_epochs}")
        print("=" * 70)

        async with state.lock:
            workers_snapshot = list(state.worker_writers.items())

        print(f"\n[Época {epoch+1}] Enviando pesos...")
        for wid, writer in workers_snapshot:
            try:
                await send_weights_to_worker(writer, state.global_weights, epoch + 1)
                print(f"   Pesos → Worker {wid}")
            except Exception as e:
                print(f"   ✗ Error enviando a Worker {wid}: {e}")

        print(f"\n[Época {epoch+1}] Esperando gradientes...")
        try:
            async with state.lock:
                ready_event = state.all_workers_ready
            await asyncio.wait_for(ready_event.wait(), timeout=600.0)

            async with state.lock:
                gradients_list         = list(state.worker_gradients.values())
                losses_list            = list(state.worker_losses.values())
                worker_losses_snapshot = dict(state.worker_losses)

            # Agregar y actualizar con momentum
            avg_grads = _average_gradients(gradients_list)
            state.global_weights, velocity = _apply_gradients_with_momentum(
                state.global_weights, avg_grads, velocity,
                lr=state.learning_rate, momentum=0.9,
            )

            avg_loss = float(np.mean(losses_list))
            test_loss, test_acc = _evaluate(state.X_test, state.y_test)

            async with state.lock:
                epoch_time = time.time() - state.epoch_start_time

            state.train_losses.append(avg_loss)
            state.test_losses.append(test_loss)
            state.test_accuracies.append(test_acc)
            state.epoch_times.append(epoch_time)

            async with state.lock:
                for wid, lv in worker_losses_snapshot.items():
                    state.worker_loss_history.setdefault(wid, []).append(lv)

            save_epoch_metrics(epoch, avg_loss, test_loss, test_acc,
                               epoch_time, worker_losses_snapshot)

            print(f"\n RESULTADOS ÉPOCA {epoch + 1}:")
            print(f"  Train loss (avg workers): {avg_loss:.4f}")
            print(f"  Test  loss:               {test_loss:.4f}")
            print(f"  Test  accuracy:           {test_acc:.4f}")
            print(f"  Tiempo época:             {epoch_time:.2f}s")

        except asyncio.TimeoutError:
            async with state.lock:
                received = len(state.worker_gradients)
                expected = len(state.worker_writers)
                missing  = set(state.worker_writers.keys()) - \
                           set(state.worker_gradients.keys())
            print(f"\n✗ Timeout época {epoch+1}: {received}/{expected} respondieron")
            print(f"  Workers faltantes: {missing}")
            break

        except Exception as e:
            print(f"\n✗ Error en época {epoch+1}: {e}")
            import traceback; traceback.print_exc()
            break

        print()

    # ── Final ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 70)
    final_loss, final_acc = _evaluate(state.X_test, state.y_test)
    total_time = time.time() - state.training_start_time
    print(f"Loss final: {final_loss:.4f} | Accuracy final: {final_acc:.4f}")
    print(f"Tiempo total: {total_time:.2f}s")
    print("=" * 70)

    save_final_summary(final_loss, final_acc, total_time)