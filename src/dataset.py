import os

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    FULL_DATASET,
    TRAIN_FILE,
    TEST_FILE,
    MAX_LENGTH,
    TEST_SIZE,
    RANDOM_STATE,
    DEV_MODE,
    DEV_TRAIN_SAMPLES,
    DEV_TEST_SAMPLES
)

# =====================================================
# Create Train/Test CSV (First Run Only)
# =====================================================

if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):

    print("\nCreating Train/Test CSV Files...\n")

    df = pd.read_csv(FULL_DATASET)

    df["label"] = df["sentiment"].map({
        "negative": 0,
        "positive": 1
    })

    df = df[["review", "label"]]

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

# =====================================================
# Load CSV Files
# =====================================================

print("\nLoading Dataset...\n")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# =====================================================
# Convert to HuggingFace Dataset
# =====================================================

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)

# =====================================================
# Development Mode
# =====================================================

if DEV_MODE:

    print("Development Mode Enabled")

    train_dataset = train_dataset.shuffle(
        seed=RANDOM_STATE
    ).select(range(DEV_TRAIN_SAMPLES))

    test_dataset = test_dataset.shuffle(
        seed=RANDOM_STATE
    ).select(range(DEV_TEST_SAMPLES))

# =====================================================
# Tokenizer
# =====================================================

print("\nLoading Tokenizer...\n")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# =====================================================
# Tokenization
# =====================================================

def tokenize_function(examples):

    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

print("Tokenizing Training Dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

print("\nTokenizing Testing Dataset...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

# =====================================================
# Rename Label
# =====================================================

train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels"
)

# =====================================================
# Remove Review Column
# =====================================================

train_dataset = train_dataset.remove_columns(
    ["review"]
)

test_dataset = test_dataset.remove_columns(
    ["review"]
)

# =====================================================
# Torch Format
# =====================================================

train_dataset.set_format("torch")
test_dataset.set_format("torch")

print("\nDataset Ready Successfully!")

print(f"Training Samples : {len(train_dataset)}")
print(f"Testing Samples  : {len(test_dataset)}")

# =====================================================
# Export
# =====================================================

train_valid = {
    "train": train_dataset,
    "test": test_dataset
}