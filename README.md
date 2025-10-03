# Sentiment analysis on product reviews

This project is an end-to-end sentiment analysis pipeline built with Python and scikit-learn.  
It classifies online product reviews into **negative, neutral, or positive** sentiments.  

---

## Project Structure

```bash
AI_Challenge/
├── notebooks/
│   ├── analysis.ipynb          # Exploratory data analysis: sentiment distribution, text length, word clouds
├── src/
│   ├── api.py                  # FastAPI app
│   ├── main.py                 # The main script (inspect, preprocess, train, test)
│   ├── inspect_data.py         # Dataset exploration & sentiment mapping
│   ├── preprocess.py           # Text preprocessing functions
│   ├── train.py                # Model training and evaluation
│   ├── predict.py              # Model loading and prediction helper
├── data/                       
│   └── reviews.csv             # Dataset file (after download)
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## Dataset

We use the **Grammar and Online Product Reviews** dataset from Kaggle:  
[Dataset link](https://www.kaggle.com/datasets/datafiniti/grammar-and-online-product-reviews?utm_source=chatgpt.com)  

After download, place the CSV file in the `data/` folder and rename it to:  
```
data/reviews.csv
```

---

## How It Works

The project follows a three-step ML pipeline:

### 1. Inspect & Preprocess Data
- Explore dataset (distribution, rating to sentiment mapping).  
- Clean reviews:
  - Lowercase  
  - Remove stopwords  
  - Lemmatize words  
- Map star ratings - sentiment:
  - 1-2 → Negative  
  - 3 → Neutral  
  - 4-5 → Positive  

### 2. Train Models
- Convert text into TF-IDF features (numerical representation).  
- Train & evaluate:
  - Logistic Regression  
  - Naive Bayes  
- Save best-performing model as:
  - `model.pkl`  
  - `vectorizer.pkl`

### 3. Predict Sentiment
- Input: raw review text
- Output: sentiment label (`positive`, `neutral`, `negative`).  

Example request:
json
```
POST /predict
{
  "text": "This laptop is fantastic, super fast and reliable!"
}
```


Response:
```json
{"sentiment": "positive"}
```

### 4. Jupyter analysis
The analysis.ipynb notebook in the notebooks/ folder contains exploratory data analysis, including:
- Sentiment distribution of reviews.
- Text length analysis.
- Word frequency and visualization (word clouds).
---

## Running the Pipeline

### 1. Data inspection & training
From the project root:
```bash
python src/main.py
```
This will:
- Inspect dataset statistics & generate plots.  
- Preprocess text.  
- Train, evaluate, and save the best model.  
- Add a few test reviews for sanity check.  

### 2. Launch API server
Run:
```bash
uvicorn src.api:app --reload
```

Open Swagger UI at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to test interactively.

### 3. Running the Jupyter analysis
Run: 
```bash
jupyter notebook
```
Open notebooks/analysis.ipynb and run the cells step by step.

## Requirements

Install all dependencies:
```bash
pip install -r requirements.txt
```
---

## Outputs
- Sentiment distribution plots.  
- Trained ML model (`model.pkl`) and TF-IDF vectorizer (`vectorizer.pkl`).
---

