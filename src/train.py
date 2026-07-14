import numpy as np
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from config import (
    MODEL_NAME,
    MODEL_DIR,
    NUM_LABELS,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE
)

from dataset import (
    train_valid,
    tokenizer
)

# =====================================================
# USE SMALL DATASET DURING DEVELOPMENT
# =====================================================

train_valid["train"] = (
    train_valid["train"]
    .shuffle(seed=42)
    .select(range(1000))
)

train_valid["test"] = (
    train_valid["test"]
    .shuffle(seed=42)
    .select(range(200))
)

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

model.to(DEVICE)

print(f"Using Device : {DEVICE}")
print(f"Training Samples : {len(train_valid['train'])}")
print(f"Testing Samples  : {len(train_valid['test'])}")

# =====================================================
# Metrics
# =====================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }

# =====================================================
# Training Arguments
# =====================================================

training_args = TrainingArguments(

    output_dir=MODEL_DIR,

    overwrite_output_dir=True,

    num_train_epochs=1,

    per_device_train_batch_size=BATCH_SIZE,

    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    eval_strategy="epoch",

    save_strategy="epoch",

    logging_strategy="steps",

    logging_steps=10,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    save_total_limit=1,

    report_to="none"
)

# =====================================================
# Trainer
# =====================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_valid["train"],

    eval_dataset=train_valid["test"],

    processing_class=tokenizer,

    compute_metrics=compute_metrics
)

print("\nStarting Training...\n")

trainer.train()

print("\nSaving Model...\n")

trainer.save_model(MODEL_DIR)

tokenizer.save_pretrained(MODEL_DIR)

print("=" * 60)
print("Training Completed Successfully!")
print("=" * 60)