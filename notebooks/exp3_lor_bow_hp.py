import os
import mlflow
import mlflow.sklearn
import joblib
import dagshub
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------- MLflow + DagsHub -----------------
mlflow.set_tracking_uri('https://dagshub.com/Vaibha3246/mlops-mini-project.mlflow')
dagshub.init(repo_owner='Vaibha3246', repo_name='mlops-mini-project', mlflow=True)

# ----------------- Load Data -----------------
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])

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

df = normalize_text(df)
df = df[df['sentiment'].isin(['happiness','sadness'])]
df['sentiment'] = df['sentiment'].replace({'sadness': 0, 'happiness': 1})

# ----------------- Train/Test Split -----------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['content'])
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------- MLflow Experiment -----------------
mlflow.set_experiment("LoR Hyperparameter Tuning")

param_grid = {
    'C': [0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# ----------------- Run MLflow -----------------
with mlflow.start_run(run_name="Parent Run") as parent_run:

    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Child runs for each parameter set
    for params, mean_score, std_score in zip(
        grid_search.cv_results_['params'],
        grid_search.cv_results_['mean_test_score'],
        grid_search.cv_results_['std_test_score']
    ):
        with mlflow.start_run(run_name=f"LR {params}", nested=True):
            model = LogisticRegression(**params).fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Log best run details in parent run
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)

    # Save & log best model in parent run
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, "best_model.pkl")
    mlflow.log_artifact("best_model.pkl", artifact_path="model")
    # mlflow.sklearn.log_model(best_model, "best_model")

    # Save vectorizer too (important for inference)
    joblib.dump(vectorizer, "vectorizer.pkl")
    mlflow.log_artifact("vectorizer.pkl", artifact_path="preprocessing")
    
    
    # 🔥 Log this code file
    mlflow.log_artifact(os.path.abspath(__file__), artifact_path="code")

    print(f"✅ Best Params: {best_params}")
    print(f"✅ Best F1 Score: {best_score}")
