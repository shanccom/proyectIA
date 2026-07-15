import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import EPOCHS, LEARNING_RATE, DEVICE
from dataset import get_dataloaders
from metrics import compute_metrics
from models import get_model


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Validation"):
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
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "efficientnet_b0", "vit", "swin", "cvt"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override EPOCHS from config.py (useful for quick tests).",
    )
    args = parser.parse_args()

    model_name = args.model
    epochs = args.epochs if args.epochs is not None else EPOCHS
    results_dir = Path("results") / model_name
    results_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, _ = get_dataloaders()
    model = get_model(model_name).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_metrics = validate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(val_metrics)

        print(
            f"Epoch {epoch:2d}/{epochs}  |  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), results_dir / "best_model.pth")
            print("  New best model saved.")

    with open(results_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Results saved in {results_dir}")


if __name__ == "__main__":
    main()
