from fastapi import FastAPI
from pydantic import BaseModel

import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

from inference import predict

app = FastAPI(
    title="DistilBERT Sentiment Analysis API",
    version="1.0.0"
)


class Review(BaseModel):
    text: str


@app.get("/")
def home():

    return {
        "message": "DistilBERT Sentiment Analysis API is running."
    }


@app.post("/predict")
def predict_review(review: Review):

    result = predict(review.text)

    return result