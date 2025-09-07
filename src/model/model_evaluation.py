import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import dagshub
import os
import joblib

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")
file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -----------------------------
# Utility functions
# -----------------------------
def load_model(file_path: str):
    with open(file_path, "rb") as file:
        return pickle.load(file)

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }

def save_metrics(metrics: dict, file_path: str) -> None:
    with open(file_path, "w") as file:
        json.dump(metrics, file, indent=4)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    model_info = {"run_id": run_id, "model_path": model_path}
    with open(file_path, "w") as file:
        json.dump(model_info, file, indent=4)

# -----------------------------
# Main execution
# -----------------------------
def main():
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("❌ DAGSHUB_PAT not found. Add it in GitHub Actions secrets.")

    os.environ["MLFLOW_TRACKING_USERNAME"] = "Vaibha3246"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Vaibha3246"
    repo_name = "mlops-mini-project"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:
        clf = load_model("./models/model.pkl")
        test_data = load_data("./data/processed/test_bow.csv")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")

        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        # Log params
        if hasattr(clf, "get_params"):
            for name, value in clf.get_params().items():
                mlflow.log_param(name, value)

        # Save + log model
        joblib.dump(clf, "model.pkl")
        mlflow.log_artifact("model.pkl", artifact_path="model")

        # Save + log model info
        save_model_info(run.info.run_id, "model", "reports/model_info.json")
        mlflow.log_artifact("reports/model_info.json")

        # Log reports + logs
        mlflow.log_artifact("reports/metrics.json")
        mlflow.log_artifact("model_evaluation_errors.log")

        # Log code itself
        mlflow.log_artifact(os.path.abspath(__file__), artifact_path="code")

        print("✅ Model evaluation and logging completed successfully.")

if __name__ == "__main__":
    main()
