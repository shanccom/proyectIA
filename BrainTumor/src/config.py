import torch

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset/BrainTumor_Dataset"
NUM_CLASSES = 2
SEED = 42

# ──────────────────────────────────────────────
# Hardware
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# Fase 1 — Entrenamiento Base
# ──────────────────────────────────────────────
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# ──────────────────────────────────────────────
# Fase 2 — Entrenamiento Optimizado
# ──────────────────────────────────────────────
USE_AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 7
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
MIXED_PRECISION = True

# Criterios de parada por convergencia
STOP_ACCURACY = 0.995        # Si val_acc ≥ 99.5% → para
CONVERGENCE_WINDOW = 5       # Ventana para detectar convergencia
MIN_DELTA = 0.001            # Mejora mínima en loss para no considerar convergido
MIN_LR = 1e-6                # Si LR baja de esto → modelo convergió
