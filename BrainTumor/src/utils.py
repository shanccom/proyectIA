import time

import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    def __init__(self):
        self.start = None

    def begin(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start

    def elapsed_str(self):
        sec = self.elapsed()
        if sec < 60:
            return f"{sec:.1f}s"
        elif sec < 3600:
            return f"{sec / 60:.1f}m"
        else:
            return f"{sec / 3600:.1f}h"


def gpu_memory_usage():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


class ConvergenceStopper:
    """Detecta cuándo el modelo ha convergido usando múltiples señales."""

    def __init__(
        self,
        patience=7,
        stop_accuracy=0.995,
        convergence_window=5,
        min_delta=0.001,
        min_lr=1e-6,
    ):
        self.patience = patience
        self.stop_accuracy = stop_accuracy
        self.convergence_window = convergence_window
        self.min_delta = min_delta
        self.min_lr = min_lr

        self.best_loss = float("inf")
        self.epochs_no_improve = 0
        self.loss_history = []

    def check(self, epoch, val_loss, val_accuracy, current_lr):
        reasons = []

        if val_accuracy >= self.stop_accuracy:
            reasons.append(f"Precisión objetivo alcanzada ({val_accuracy:.4f} ≥ {self.stop_accuracy})")

        if current_lr < self.min_lr:
            reasons.append(f"LR mínimo alcanzado ({current_lr:.2e} < {self.min_lr})")

        self.loss_history.append(val_loss)
        if len(self.loss_history) >= self.convergence_window:
            recent = self.loss_history[-self.convergence_window:]
            relative_change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-8)
            if relative_change < self.min_delta:
                reasons.append(f"Convergencia detectada (cambio relativo {relative_change:.6f} < {self.min_delta})")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            reasons.append(f"Early stopping ({self.epochs_no_improve} épocas sin mejora)")

        return reasons
