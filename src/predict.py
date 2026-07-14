from inference import predict

print("=" * 60)
print("DistilBERT Sentiment Analyzer")
print("Type 'exit' to quit.")
print("=" * 60)

while True:

    text = input("\nEnter Review : ")

    if text.lower() == "exit":
        break

    result = predict(text)

    print("\nPrediction")
    print("-" * 25)
    print(f"Sentiment : {result['sentiment']}")
    print(f"Confidence: {result['confidence']}%")