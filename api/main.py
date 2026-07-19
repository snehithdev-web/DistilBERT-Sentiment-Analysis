print("main.py started")
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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