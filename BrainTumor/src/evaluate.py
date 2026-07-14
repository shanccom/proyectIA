import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

from config import DEVICE
from dataset import get_dataloaders
from metrics import compute_metrics
from models import get_model


def plot_confusion_matrix(y_true, y_pred, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained brain-tumor classification model."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "efficientnet_b0", "vit", "swin", "cvt"],
        help="Model architecture to evaluate.",
    )
    args = parser.parse_args()

    model_name = args.model
    results_dir = Path("results") / model_name

    _, _, test_loader = get_dataloaders()

    model = get_model(model_name).to(DEVICE)
    state_dict = torch.load(results_dir / "best_model.pth",
                            map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test metrics for {model_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    plot_confusion_matrix(all_labels, all_preds,
                          results_dir / "confusion_matrix.png")
    plot_roc_curve(all_labels, all_probs,
                   results_dir / "roc_curve.png")

    print(f"Results saved in {results_dir}")


if __name__ == "__main__":
    main()
