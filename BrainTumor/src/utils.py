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
    """Detecta convergencia usando Validation Loss (early stopping) y LR mínimo."""

    def __init__(self, patience=7, min_lr=1e-6):
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float("inf")
        self.epochs_no_improve = 0

    def check(self, epoch, val_loss, val_accuracy, current_lr):
        reasons = []

        if current_lr < self.min_lr:
            reasons.append(f"LR mínimo alcanzado ({current_lr:.2e} < {self.min_lr})")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            reasons.append(f"Early stopping ({self.epochs_no_improve} épocas sin mejora en val_loss)")

        return reasons
