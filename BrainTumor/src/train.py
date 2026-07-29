import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import (
    EPOCHS, LEARNING_RATE, WEIGHT_DECAY, DEVICE, SEED, NUM_CLASSES,
    EARLY_STOPPING_PATIENCE, LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE, MIXED_PRECISION, USE_AUGMENTATION, MIN_LR,
)
from dataset import get_dataloaders
from metrics import compute_metrics
from models import get_model, print_model_info
from utils import Timer, gpu_memory_usage, ConvergenceStopper


def train_one_epoch(model, loader, criterion, optimizer, scaler, use_amp):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="  Val"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a brain-tumor classification model."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["resnet50", "efficientnet_b0", "vit", "swin", "coatnet"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Fase 1 = Base (sin optimización), Fase 2 = Optimizado.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override EPOCHS from config.py (límite máximo).",
    )
    parser.add_argument(
        "--scratch", action="store_true", default=False,
        help="Entrenar desde cero sin pesos preentrenados.",
    )
    parser.add_argument(
        "--normalized", action="store_true", default=False,
        help="Usar pipeline de normalizacion estricta (elimina artefactos).",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = args.model
    phase = args.phase
    is_optimized = phase == 2
    max_epochs = args.epochs if args.epochs is not None else EPOCHS
    use_amp = MIXED_PRECISION and is_optimized and DEVICE == "cuda"
    use_augment = USE_AUGMENTATION and is_optimized

    scratch = args.scratch
    normalized = args.normalized
    if scratch and normalized:
        phase_tag = "normalized_scratch"
        is_optimized = False
        use_augment = False
        use_amp = False
    elif scratch:
        phase_tag = "scratch"
        is_optimized = False
        use_augment = False
        use_amp = False
    elif normalized:
        phase_tag = "normalized"
        is_optimized = True
    else:
        phase_tag = "optimized" if is_optimized else "base"
    results_dir = Path("results") / model_name / phase_tag
    results_dir.mkdir(parents=True, exist_ok=True)

    stopper = ConvergenceStopper(
        patience=EARLY_STOPPING_PATIENCE if is_optimized else 999,
        min_lr=MIN_LR if is_optimized else 0.0,
    )

    print(f"\nModelo:     {model_name}")
    if scratch:
        print(f"Fase:       Scratch (sin pesos preentrenados)")
    elif normalized:
        print(f"Fase:       Normalizado (pipeline anti-artefactos)")
    else:
        print(f"Fase:       {phase} ({'Optimizado' if is_optimized else 'Base'})")
    print(f"Epocas max: {max_epochs}")
    print(f"Augment:    {'Si' if use_augment else 'No'}")
    print(f"AMP:        {'Si' if use_amp else 'No'}")
    print(f"Normalized: {'Si (sin artefactos)' if normalized else 'No'}")
    print(f"Device:     {DEVICE}")
    print(f"Seed:       {SEED}")
    print(f"Pretrained: {'Si' if not scratch else 'No (desde cero)'}")
    print(f"Parada por:")
    if is_optimized:
        print(f"  - Early stopping (paciencia={EARLY_STOPPING_PATIENCE} en val_loss)")
        print(f"  - LR minimo < {MIN_LR}")
        print(f"  - Maximo de epocas ({max_epochs})")
    else:
        print(f"  - Maximo de epocas ({max_epochs})")
    print()

    train_loader, val_loader, _ = get_dataloaders(augment=use_augment,
                                                    normalized=normalized)
    model = get_model(model_name, pretrained=not scratch).to(DEVICE)
    model_info = print_model_info(model, model_name)

    class_counts = np.bincount(train_loader.dataset.targets)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * NUM_CLASSES
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY if is_optimized else 0.0)

    scheduler = None
    if is_optimized:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE
        )

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    history = {
        "train_loss": [], "train_acc": [], "val_loss": [],
        "val_metrics": [], "lr": [], "epoch_time": [],
    }
    best_val_loss = float("inf")
    timer = Timer()
    stop_reason = "Máximo de épocas alcanzado"

    for epoch in range(1, max_epochs + 1):
        timer.begin()
        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        print(f"Epoch {epoch:2d}/{max_epochs}  |  LR: {current_lr:.2e}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                optimizer, scaler, use_amp)
        val_loss, val_metrics = validate(model, val_loader, criterion)

        epoch_time = timer.elapsed()
        history["epoch_time"].append(epoch_time)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        print(
            f"  -> Train Loss: {train_loss:.4f}  |  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val Acc: {val_metrics['accuracy']:.4f}  |  "
            f"Tiempo: {timer.elapsed_str()}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print("  [OK] Nuevo mejor modelo guardado.")

        reasons = stopper.check(
            epoch, val_loss, val_metrics["accuracy"], current_lr
        )
        if reasons:
            for r in reasons:
                print(f"  [STOP] {r}")
            stop_reason = "; ".join(reasons)
            break

    history["epochs_executed"] = epoch
    history["stop_reason"] = stop_reason
    model_info["gpu_memory_gb"] = gpu_memory_usage()

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(results_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"\nEntrenamiento finalizado - {stop_reason}")
    print(f"Epocas ejecutadas: {epoch}")
    print(f"Resultados en: {results_dir}")


if __name__ == "__main__":
    main()
