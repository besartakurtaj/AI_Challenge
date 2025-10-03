from inspect_data import inspect_dataset
from preprocess import preprocess_dataset, preprocess_text
from train import train_models
from predict import load_model, predict_sentiment

if __name__ == "__main__":
    print("First step: Inspecting dataset...")
    df = inspect_dataset()

    print("\n Second step: Preprocessing dataset..")
    df = preprocess_dataset(df)

    print("\n Third step: Training models...")
    train_models(df)
    
    print("\n Last step: Loading best model and making predictions...\n")
    model, vectorizer = load_model()

    demo_reviews = [
        "This product is terrible, broke in 2 days!",
        "It's okay, not too bad but not great either.",
        "Absolutely fantastic!"
    ]
    for review in demo_reviews:
        cleaned_review = preprocess_text(review)
        print(f"Review: {review}")
        print(f"Preprocessed: {cleaned_review}")
        print("Predicted Sentiment:", predict_sentiment(cleaned_review, model, vectorizer))
        print("-"*50)