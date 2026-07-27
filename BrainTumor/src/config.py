import torch

# Dataset
IMAGE_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset/BrainTumor_Dataset"
NUM_CLASSES = 2
SEED = 42

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fase 1 — Entrenamiento Base
EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Fase 2 — Entrenamiento Optimizado
USE_AUGMENTATION = True
EARLY_STOPPING_PATIENCE = 7
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 3
MIXED_PRECISION = True

MIN_LR = 1e-6
