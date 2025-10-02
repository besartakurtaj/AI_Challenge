import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def inspect_dataset(path="../data/reviews.csv"):
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print("\nUnique ratings:", df["reviews.rating"].unique())
    print("Ratings distribution:\n", df["reviews.rating"].value_counts())

    def map_rating(r):
        if r <= 2: return "negative"
        elif r == 3: return "neutral"
        else: return "positive"

    df = df[["reviews.text", "reviews.rating"]].dropna()
    df["sentiment"] = df["reviews.rating"].apply(map_rating)

    print("\nSentiment distribution:\n", df["sentiment"].value_counts())

    # save plot
    sns.countplot(data=df, x="sentiment", order=["negative", "neutral", "positive"])
    plt.title("Sentiment Distribution")
    plt.savefig("sentiment_distribution.png")
    plt.close()

    return df
