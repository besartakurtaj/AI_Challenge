import sys, os
sys.path.append(os.path.dirname(__file__))  #ensures src/ is always on sys.path

from fastapi import FastAPI
from pydantic import BaseModel
from predict import load_model, predict_sentiment

app = FastAPI()
model, vectorizer = load_model()

class ReviewInput(BaseModel):
    text: str

@app.post("/predict")
def predict_endpoint(review: ReviewInput):
    sentiment = predict_sentiment(review.text, model, vectorizer)
    return {"sentiment": sentiment}
