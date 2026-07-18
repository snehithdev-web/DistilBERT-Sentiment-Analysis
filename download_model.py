from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL_NAME)

print("Downloading model...")
AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

print("Done")