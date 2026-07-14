import os
import torch

# ==========================================
# Project Paths
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")

TEST_FILE = os.path.join(DATA_DIR, "test.csv")

FULL_DATASET = os.path.join(DATA_DIR, "IMDB Dataset.csv")

# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256

NUM_LABELS = 2

# ==========================================
# Training Configuration
# ==========================================

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

NUM_EPOCHS = 3

WEIGHT_DECAY = 0.01

TEST_SIZE = 0.20

RANDOM_STATE = 42

# ==========================================
# Device
# ==========================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)