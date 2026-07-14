# Brain Tumor Classification

Single pipeline for training and evaluating multiple CNN / Transformer
architectures on brain-tumor MRI data.

```
python src/train.py --model resnet50
python src/train.py --model efficientnet_b0
python src/train.py --model vit
python src/train.py --model swin
python src/train.py --model cvt

python src/evaluate.py --model resnet50
```

## Project structure

```
BrainTumor/
├── dataset/         # ImageFolder with class sub-folders
├── src/
│   ├── config.py    # Hyperparameters & paths
│   ├── dataset.py   # Data-loading pipeline
│   ├── metrics.py   # Accuracy, Precision, Recall, F1, ROC AUC
│   ├── models.py    # Model factory (ResNet, EfficientNet, ViT, Swin, CVT)
│   ├── train.py     # Training loop
│   ├── evaluate.py  # Test evaluation + plots
│   └── utils.py     # Helpers
├── notebooks/       # Jupyter notebooks (exploration / analysis)
├── results/         # Per-model results
│   ├── resnet50/
│   ├── efficientnet/
│   ├── vit/
│   ├── swin/
│   └── cvt/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place your MRI dataset in `dataset/` with an ImageFolder layout:

```
dataset/
├── tumor/
└── no_tumor/
```
