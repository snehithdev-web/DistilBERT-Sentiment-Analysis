import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader

from config import (
    MODEL_DIR,
    DEVICE,
    BATCH_SIZE
)

from dataset import train_valid

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()

test_loader = DataLoader(
    train_valid["test"],
    batch_size=BATCH_SIZE,
    shuffle=False
)

correct = 0
total = 0

print("\nEvaluating Model...\n")

with torch.no_grad():

    for batch in tqdm(test_loader):

        input_ids = batch["input_ids"].to(DEVICE)

        attention_mask = batch["attention_mask"].to(DEVICE)

        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

accuracy = correct / total

print("\n" + "=" * 60)
print(f"Accuracy : {accuracy:.4f}")
print("=" * 60)