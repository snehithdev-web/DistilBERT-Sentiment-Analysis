import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from config import (
    MODEL_DIR,
    DEVICE
)

from dataset import train_valid

# ---------------------------------------
# Use same subset as training
# ---------------------------------------

test_dataset = (
    train_valid["test"]
    .shuffle(seed=42)
    .select(range(200))
)

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

correct = 0
total = 0

print("\nEvaluating Model...\n")

with torch.no_grad():

    for sample in tqdm(test_dataset):

        input_ids = sample["input_ids"].unsqueeze(0).to(DEVICE)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(DEVICE)

        label = sample["labels"].item()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        prediction = outputs.logits.argmax(dim=1).item()

        if prediction == label:
            correct += 1

        total += 1

accuracy = correct / total

print("\n" + "=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print("=" * 60)