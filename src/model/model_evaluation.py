import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
import joblib

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")

if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "Vaibha3246"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    print("✅ Using DAGSHUB_PAT for authentication.")
else:
    print("⚠️ DAGSHUB_PAT not found. Proceeding without authentication.")

# Set up MLflow tracking URI


dagshub_url = "https://dagshub.com"
repo_owner = "Vaibha3246"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# logging configuration
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

def load_model(file_path: str):
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }
        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving metrics: %s", e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error saving model info: %s", e)
        raise

def main():
    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)

            save_metrics(metrics, "reports/metrics.json")

            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log params
            if hasattr(clf, "get_params"):
                for param_name, param_value in clf.get_params().items():
                    mlflow.log_param(param_name, param_value)

            # Save and log model
            joblib.dump(clf, "model.pkl")
            mlflow.log_artifact("model.pkl", artifact_path="model")

            # Save and log model info (FIXED: consistent filename)
            save_model_info(run.info.run_id, "model", "reports/model_info.json")
            mlflow.log_artifact("reports/model_info.json")

            # Log metrics file
            mlflow.log_artifact("reports/metrics.json")

            # Log error log file
            mlflow.log_artifact("model_evaluation_errors.log")
            
            # ✅ Log the code file itself
            current_file = os.path.abspath(__file__)
            mlflow.log_artifact(current_file, artifact_path="code")

        except Exception as e:
            logger.error("Failed to complete the model evaluation process: %s", e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
