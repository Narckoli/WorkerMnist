# Export_Metrics.py - Servidor
"""
Sistema de archivos de métricas:

  training_results/
  ├── experiments_summary.csv        ← UNA FILA por experimento (acumulativo)
  └── epochs/
      └── cifar10_cnn_3w_20260404_165727.csv  ← detalle de épocas del experimento

El archivo de épocas se crea una sola vez por experimento (el timestamp
se fija al arrancar save_epoch_metrics por primera vez).
"""

import csv
import os
from datetime import datetime
from Config import state

_RESULTS_DIR = "training_results"
_EPOCHS_DIR  = os.path.join(_RESULTS_DIR, "epochs")

# Fijado en la primera llamada a save_epoch_metrics — no cambia entre épocas
_experiment_ts   = None
_epochs_filepath = None


def _init_experiment():
    """Inicializa timestamp y ruta del archivo de épocas (solo una vez)."""
    global _experiment_ts, _epochs_filepath
    if _experiment_ts is not None:
        return  # ya inicializado

    os.makedirs(_EPOCHS_DIR, exist_ok=True)
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    _experiment_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"{state.dataset_name}_{state.model_type}_"
            f"{state.expected_workers}w_{_experiment_ts}.csv")
    _epochs_filepath = os.path.join(_EPOCHS_DIR, name)


def save_epoch_metrics(epoch: int, avg_loss: float, test_loss: float,
                       test_acc: float, epoch_time: float,
                       worker_losses: dict) -> str | None:
    """
    Guarda las métricas de una época en el archivo de detalle.
    En la época 0 escribe la cabecera y los metadatos del experimento.
    """
    _init_experiment()
    is_first = (epoch == 0)

    try:
        with open(_epochs_filepath, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)

            if is_first:
                # Metadatos
                w.writerow(["# dataset",      state.dataset_name])
                w.writerow(["# model_type",   state.model_type])
                w.writerow(["# workers",      state.expected_workers])
                w.writerow(["# epochs",       state.max_epochs])
                w.writerow(["# learning_rate",state.learning_rate])
                w.writerow(["# input_size",   state.input_size])
                w.writerow(["# timestamp",    _experiment_ts])
                w.writerow([])

                # Cabecera de datos
                header = ["epoch", "train_loss", "test_loss",
                          "test_accuracy", "epoch_time_s"]
                if worker_losses:
                    header += [f"worker_{wid}_loss"
                               for wid in sorted(worker_losses)]
                w.writerow(header)

            # Fila de datos
            row = [epoch + 1,
                   f"{avg_loss:.6f}",
                   f"{test_loss:.6f}",
                   f"{test_acc:.6f}",
                   f"{epoch_time:.3f}"]
            if worker_losses:
                for wid in sorted(worker_losses):
                    row.append(f"{worker_losses[wid]:.6f}")
            w.writerow(row)

        if is_first:
            print(f"✓ Archivo de épocas: {_epochs_filepath}")
        return _epochs_filepath

    except Exception as e:
        print(f"✗ Error guardando métricas de época: {e}")
        return None


def save_final_summary(final_loss: float, final_acc: float,
                       total_time: float) -> str | None:
    """
    Agrega UNA fila al CSV acumulativo de experimentos.
    Si el archivo no existe lo crea con cabecera.
    """
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    summary_file = os.path.join(_RESULTS_DIR, "experiments_summary.csv")
    file_exists  = os.path.isfile(summary_file)

    # Mejor accuracy y época correspondiente
    best_acc   = max(state.test_accuracies) if state.test_accuracies else final_acc
    best_epoch = (state.test_accuracies.index(best_acc) + 1
                  if state.test_accuracies else "-")

    data = {
        "timestamp":           datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset":             state.dataset_name,
        "model_type":          state.model_type,
        "workers":             state.expected_workers,
        "epochs":              state.max_epochs,
        "learning_rate":       state.learning_rate,
        "input_size":          state.input_size,
        "total_time_seconds":  f"{total_time:.1f}",
        "total_time_minutes":  f"{total_time / 60:.2f}",
        "final_test_loss":     f"{final_loss:.6f}",
        "final_test_accuracy": f"{final_acc:.6f}",
        "best_test_accuracy":  f"{best_acc:.6f}",
        "best_epoch":          best_epoch,
        "epochs_file":         _epochs_filepath or "",
    }

    try:
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                w.writeheader()
                print(f"✓ Creado archivo de resumen: {summary_file}")
            w.writerow(data)
        print(f"✓ Resumen agregado a: {summary_file}")
        return summary_file
    except Exception as e:
        print(f"✗ Error exportando resumen: {e}")
        return None