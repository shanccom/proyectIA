import torch

# Data
IMAGE_SIZE = 224
BATCH_SIZE = 32
DATASET_PATH = "dataset/BrainTumor_Dataset"
NUM_CLASSES = 2
SEED = 42

# Training
EPOCHS = 30
LEARNING_RATE = 1e-4

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
