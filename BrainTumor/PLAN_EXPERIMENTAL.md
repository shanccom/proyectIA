# Plan Experimental — Clasificación de Tumores Cerebrales

## 1. Framework

**PyTorch + torchvision** (modelos pre-entrenados de `torchvision.models`).

| Modelo | Tipo | Pesos |
|---|---|---|
| ResNet50 | CNN | `ResNet50_Weights.DEFAULT` |
| EfficientNet-B0 | CNN | `EfficientNet_B0_Weights.DEFAULT` |
| ViT-B/16 | Transformer | `ViT_B_16_Weights.DEFAULT` |
| Swin-T | Transformer | `Swin_T_Weights.DEFAULT` |
| CvT | Híbrido CNN+Transformer | `cvt-pytorch` |

Estrategia: **Transfer Learning** — backbone pre-entrenado en ImageNet, reemplazar head, fine-tune completo.

## 2. Hiperparámetros

| Parámetro | Valor |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Épocas | 30 |
| Optimizer | Adam |
| Learning rate | 1×10⁻⁴ |
| Loss | CrossEntropy |
| Seed | 42 |
| Device | CUDA (auto-detectado) |

## 3. Dataset

- **Split**: ya pre-dividido (`train/` `val/` `test/`)
- **Clases**: `no` (0) y `yes` (1)
- **Distribución**:
  - Train: 5609 (no: 2295, yes: 3314)
  - Val:   1403 (no: 574,  yes: 829)
  - Test:  1752 (no: 717,  yes: 1035)
- **Preprocesamiento**: Resize(224) → ToTensor → Normalize(ImageNet)
- **Data augmentation** (solo train): RandomHorizontalFlip, RandomRotation(10°)

## 4. Métricas (para todos los modelos)

Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.

## 5. Pipeline

```
python src/train.py --model resnet50 [--epochs N]
python src/evaluate.py --model resnet50
```

Cada modelo genera en `results/<modelo>/`:
- `best_model.pth` — pesos del mejor checkpoint
- `history.json` — loss y métricas por época
- `metrics.json` — métricas finales en test
- `confusion_matrix.png` — matriz de confusión
- `roc_curve.png` — curva ROC

## 6. Distribución del trabajo (4 integrantes — 5 modelos)

| Integrante | Modelos | Días estimados (Colab GPU) |
|---|---|---|
| **A** | ResNet50 + EfficientNet-B0 | ~2 h c/u = 4 h |
| **B** | ViT-B/16 | ~3 h |
| **C** | Swin-T | ~2 h |
| **D** | CvT | ~2 h |

Los integrantes B, C, D también pueden ayudar a evaluar los modelos de A al terminar.

**Flujo de trabajo**:
1. Cada uno clona el repo y sube el dataset a su Drive
2. Abre `notebooks/colab_setup.ipynb` en Colab
3. Cambia `--model` por el suyo y ejecuta
4. Al terminar, copia `results/<modelo>/` a una carpeta compartida (Drive/GitHub)
5. El integrante A unifica todos los `results/` y ejecuta la comparación final

## 7. Checklist de verificacion

| Item | Estado |
|---|---|
| Dataset descargado y organizado | Si |
| Division pre-hecha (train/val/test) | Si |
| Dependencias instaladas | Si (torch 2.13, torchvision 0.28, matplotlib, sklearn, tqdm) |
| requirements.txt fijado | Si |
| Config experimental definida | Si (config.py) |
| Metricas definidas | Si (metrics.py) |
| Pipeline de entrenamiento | Si (train.py + argparse) |
| Pipeline de evaluacion | Si (evaluate.py) |
| Colab setup listo | Si (notebooks/colab_setup.ipynb) |
| Flag --epochs para pruebas | Si |
| Auto-detectar CPU / CUDA | Si |
| Data augmentation en train | Si (flip + rotation) |
| Carpetas de resultados creadas | Si |
| Estructura subida a git | Si (.gitkeep en carpetas vacias) |

## 8. Pendiente — antes del entrenamiento final

- Verificar que no hay imágenes corruptas en el dataset
- Decidir versión de CvT (requiere `pip install cvt-pytorch` en Colab)
- Hacer una prueba de 2 épocas cada uno para confirmar que no falla nada

## 9. Comandos rápidos

```bash
# Prueba de 2 épocas (tarda ~2 min en Colab GPU)
python src/train.py --model resnet50 --epochs 2

# Entrenamiento completo
python src/train.py --model resnet50

# Evaluar
python src/evaluate.py --model resnet50
```
