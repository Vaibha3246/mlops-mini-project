# Bow vs TF-IDF Experiment with MLflow + DagsHub

import os
import re
import string
import joblib
import mlflow
import mlflow.sklearn
import dagshub
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Allow insecure TLS for DagsHub MLflow
os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

# Connect MLflow to DagsHub
mlflow.set_tracking_uri('https://dagshub.com/Vaibha3246/mlops-mini-project.mlflow')
dagshub.init(repo_owner='Vaibha3246', repo_name='mlops-mini-project', mlflow=True)

# ----------------- Load Data -----------------
df = pd.read_csv(
    'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
).drop(columns=['tweet_id'])

# ----------------- Text Preprocessing -----------------
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in str(text).split() if word not in stop_words])

def removing_numbers(text):
    return ''.join([char for char in text if not char.isdigit()])

def lower_case(text):
    return " ".join([word.lower() for word in text.split()])

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    return re.sub(r'\s+', ' ', text).strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    df['content'] = df['content'].apply(lower_case)
    df['content'] = df['content'].apply(remove_stop_words)
    df['content'] = df['content'].apply(removing_numbers)
    df['content'] = df['content'].apply(removing_punctuations)
    df['content'] = df['content'].apply(removing_urls)
    df['content'] = df['content'].apply(lemmatization)
    return df

# ----------------- Prepare Data -----------------
df = normalize_text(df)

# Keep only happiness & sadness
df = df[df['sentiment'].isin(['happiness', 'sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# ----------------- Define Experiments -----------------
mlflow.set_experiment("Bow vs TfIdf")

vectorizers = {
    'BoW': CountVectorizer(),
    'TF-IDF': TfidfVectorizer()
}

algorithms = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# ----------------- MLflow Logging -----------------
with mlflow.start_run(run_name="All Experiments") as parent_run:
    results = []  # collect child results

    for algo_name, algorithm in algorithms.items():
        for vec_name, vectorizer in vectorizers.items():
            with mlflow.start_run(run_name=f"{algo_name} with {vec_name}", nested=True):
                # Feature extraction
                X = vectorizer.fit_transform(df['content'])
                y = df['sentiment']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train model
                model = algorithm
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)

                # Log metrics
                mlflow.log_param("vectorizer", vec_name)
                mlflow.log_param("algorithm", algo_name)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Save child run result
                results.append((algo_name, vec_name, accuracy, precision, recall, f1, model, vectorizer))

                print(f"Algorithm: {algo_name}, Vectorizer: {vec_name}, Acc: {accuracy:.4f}")

    # ----------------- Parent Run Aggregation -----------------
    best = max(results, key=lambda x: x[2])  # choose by accuracy
    best_algo, best_vec, best_acc, best_prec, best_rec, best_f1, best_model, best_vectorizer = best

    # Log best results in parent run
    mlflow.log_param("best_algorithm", best_algo)
    mlflow.log_param("best_vectorizer", best_vec)
    mlflow.log_metric("best_accuracy", best_acc)
    mlflow.log_metric("best_precision", best_prec)
    mlflow.log_metric("best_recall", best_rec)
    mlflow.log_metric("best_f1_score", best_f1)

    # Save best model
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl", artifact_path="model")

    # Save code
    mlflow.log_artifact(os.path.abspath(__file__), artifact_path="code")

    print(f"\n✅ Best Model: {best_algo} with {best_vec} → Acc: {best_acc:.4f}")
