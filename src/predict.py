import sys
import pickle
from preprocess import preprocess_text

def load_model():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    return model.predict(vec)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py 'add your review text here'")
        sys.exit(1)

    model, vectorizer = load_model()
    text = " ".join(sys.argv[1:])
    prediction = predict_sentiment(text, model, vectorizer)
    print("Review:", text)
    print("Predicted Sentiment:", prediction)
