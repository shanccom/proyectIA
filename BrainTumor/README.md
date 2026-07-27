# Brain Tumor Classification — Estudio Comparativo

**5 arquitecturas** de Deep Learning para clasificación binaria de tumores
cerebrales en MRI, entrenadas bajo **2 fases experimentales**.

## Comandos

```bash
# Fase 1 — Entrenamiento Base (comparación justa)
python src/train.py --model resnet50 --phase 1

# Fase 2 — Entrenamiento Optimizado (augment, early stopping, scheduler, AMP)
python src/train.py --model resnet50 --phase 2

# Evaluar
python src/evaluate.py --model resnet50 --phase 2
```

## Project structure

```
BrainTumor/
├── dataset/            # ImageFolder con train/val/test
├── src/
│   ├── config.py       # Hiperparámetros Fase 1 y 2
│   ├── dataset.py      # Data loaders con/sin augmentation
│   ├── metrics.py      # Acc, Prec, Rec, F1, ROC-AUC, Spec, Sens, MCC
│   ├── models.py       # 5 modelos (ResNet, EfficientNet, ViT, Swin, CoAtNet)
│   ├── train.py        # Entrenamiento Fase 1/2
│   ├── evaluate.py     # Evaluación + métricas computacionales
│   ├── clean_dataset.py# Validación de imágenes
│   └── utils.py        # Timer, param count, GPU memory
├── notebooks/          # colab_setup.ipynb (con EDA incluido)
├── results/
│   ├── resnet50/{base,optimized}/
│   ├── efficientnet_b0/{base,optimized}/
│   ├── vit/{base,optimized}/
│   ├── swin/{base,optimized}/
│   └── coatnet/{base,optimized}/
├── PLAN_MAESTRO.md     # Organización del equipo
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

El dataset debe estar en `dataset/BrainTumor_Dataset/` con estructura:

```
BrainTumor_Dataset/
├── train/{no,yes}/
├── val/{no,yes}/
└── test/{no,yes}/
```
