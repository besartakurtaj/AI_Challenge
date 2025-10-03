import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def train_models(df):
    # use preprocessed reviews that are already cleaned
    X_train, X_test, y_train, y_test = train_test_split(
        df["reviews.text"], df["sentiment"], test_size=0.2, random_state=42
    )

    #TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    #models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB()
    }
    #best model
    best_model, best_score = None, 0
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        print(classification_report(y_test, preds))

        cm = confusion_matrix(y_test, preds, labels=["negative","neutral","positive"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["neg","neu","pos"],
                    yticklabels=["neg","neu","pos"])
        plt.title(f"Confusion Matrix - {name}")
        plt.savefig(f"cm_{name.replace(' ','_')}.png")
        plt.close()

        acc = (preds == y_test).mean()
        if acc > best_score:
            best_model, best_score = model, acc

    #save best model and vectorizer
    pickle.dump(best_model, open("model.pkl", "wb"))
    pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

    print(f"Best model: {best_model.__class__.__name__} (Acc: {best_score:.2f})")
    return best_model, vectorizer
