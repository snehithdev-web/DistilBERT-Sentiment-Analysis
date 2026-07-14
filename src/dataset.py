import pandas as pd

from datasets import Dataset
from transformers import AutoTokenizer

from config import (
    MODEL_NAME,
    TRAIN_FILE,
    TEST_FILE,
    MAX_LENGTH,
)

# ---------------------------------------------------------
# Load Train & Test CSV Files
# ---------------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Training Samples : {len(train_df)}")
print(f"Testing Samples  : {len(test_df)}")

# ---------------------------------------------------------
# Convert Pandas DataFrame -> Hugging Face Dataset
# ---------------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)

# ---------------------------------------------------------
# Load Tokenizer
# ---------------------------------------------------------

print("\nLoading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

# ---------------------------------------------------------
# Tokenization Function
# ---------------------------------------------------------

def tokenize_function(examples):

    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

# ---------------------------------------------------------
# Tokenize Dataset
# ---------------------------------------------------------

print("\nTokenizing Training Dataset...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
)

print("\nTokenizing Testing Dataset...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
)

# ---------------------------------------------------------
# Rename label -> labels
# ---------------------------------------------------------

train_dataset = train_dataset.rename_column(
    "label",
    "labels",
)

test_dataset = test_dataset.rename_column(
    "label",
    "labels",
)

# ---------------------------------------------------------
# Remove review column
# ---------------------------------------------------------

train_dataset = train_dataset.remove_columns(
    ["review"]
)

test_dataset = test_dataset.remove_columns(
    ["review"]
)

# ---------------------------------------------------------
# Convert Dataset to PyTorch Format
# ---------------------------------------------------------

train_dataset.set_format("torch")

test_dataset.set_format("torch")

# ---------------------------------------------------------
# Export
# ---------------------------------------------------------

train_valid = {
    "train": train_dataset,
    "test": test_dataset,
}

print("\nDataset Ready Successfully!")
print("=" * 60)