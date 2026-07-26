import argparse
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import matplotlib.pyplot as plt

from config import DEVICE, SEED
from dataset import get_dataloaders
from metrics import compute_metrics
from models import get_model, print_model_info
from utils import count_parameters


def plot_confusion_matrix(y_true, y_pred, save_path):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.figure_.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained brain-tumor classification model."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["resnet50", "efficientnet_b0", "vit", "swin", "coatnet"],
        help="Model architecture to evaluate.",
    )
    parser.add_argument(
        "--phase", type=int, default=2, choices=[1, 2],
        help="Fase del entrenamiento (1=base, 2=optimizado).",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model_name = args.model
    phase_tag = "optimized" if args.phase == 2 else "base"
    results_dir = Path("results") / model_name / phase_tag

    if not (results_dir / "best_model.pth").exists():
        print(f"Error: no se encontró {results_dir / 'best_model.pth'}")
        print("Ejecuta primero el entrenamiento.")
        return

    _, _, test_loader = get_dataloaders(augment=False)

    model = get_model(model_name).to(DEVICE)
    state_dict = torch.load(results_dir / "best_model.pth",
                            map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    total_params = count_parameters(model)
    print(f"\nEvaluando: {model_name} ({phase_tag})")
    print(f"Parametros totales: {total_params:,}\n")

    all_preds, all_labels, all_probs = [], [], []

    # Medir tiempo de inferencia
    torch.cuda.empty_cache()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if DEVICE == "cuda":
                starter.record()
                outputs = model(images)
                ender.record()
                torch.cuda.synchronize()
                timings.append(starter.elapsed_time(ender))
            else:
                t0 = time.time()
                outputs = model(images)
                timings.append((time.time() - t0) * 1000)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["total_params"] = total_params

    inferencia_ms = sum(timings) / len(timings)
    metrics["inference_time_ms_per_batch"] = round(inferencia_ms, 2)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Métricas de {model_name} ({phase_tag}):")
    print("-" * 30)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")
    print(f"\nTiempo inferencia/batch: {inferencia_ms:.1f} ms")

    plot_confusion_matrix(all_labels, all_preds,
                          results_dir / "confusion_matrix.png")
    plot_roc_curve(all_labels, all_probs,
                   results_dir / "roc_curve.png")
    print(f"\nGráficas guardadas en {results_dir}")


if __name__ == "__main__":
    main()
