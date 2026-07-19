import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import (
    HF_MODEL_ID,
    DEVICE
)

# =====================================================
# Load Tokenizer
# =====================================================

print("Loading Tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    HF_MODEL_ID
)

# =====================================================
# Load Model
# =====================================================

print("Loading Model...")

model = AutoModelForSequenceClassification.from_pretrained(
    HF_MODEL_ID
)

model.to(DEVICE)

model.eval()

print("Inference Engine Ready!")

# =====================================================
# Prediction Function
# =====================================================

def predict(text: str):

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        padding=True,

        max_length=256
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence = probabilities.max().item()

    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    return {
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2)
    }