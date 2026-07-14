import os
import torch

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_DIR = os.path.join(BASE_DIR, "models")

FULL_DATASET = os.path.join(DATA_DIR, "IMDB Dataset.csv")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")

TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# =====================================================
# Model
# =====================================================

MODEL_NAME = "distilbert-base-uncased"

NUM_LABELS = 2

MAX_LENGTH = 256

# =====================================================
# Dataset
# =====================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42

# =====================================================
# Development Mode
# =====================================================

DEV_MODE = True

DEV_TRAIN_SAMPLES = 1000

DEV_TEST_SAMPLES = 200

# =====================================================
# Training
# =====================================================

NUM_EPOCHS = 3

BATCH_SIZE = 16

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

# =====================================================
# Device
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)