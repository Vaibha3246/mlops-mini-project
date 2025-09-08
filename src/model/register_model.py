import json
import mlflow
import mlflow.sklearn
import logging
import os
import glob
import joblib
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "Vaibha3246"
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "Vaibha3246"
repo_name = "mlops-mini-project"

# Set MLflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("model_registration_errors.log")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# -----------------------------
# Load model info
# -----------------------------
def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            model_info = json.load(f)
        logger.debug(f"Model info loaded from {file_path}")
        return model_info
    except Exception as e:
        logger.error(f"Failed to load model info: {e}")
        raise

# -----------------------------
# Get latest model file
# -----------------------------
def get_latest_model_file() -> str:
    search_dirs = ["src/model", "."]
    pkl_files = []

    for d in search_dirs:
        files = glob.glob(os.path.join(d, "*.pkl"))
        pkl_files.extend(files)

    if not pkl_files:
        raise FileNotFoundError("No .pkl model files found!")

    latest_model = max(pkl_files, key=os.path.getctime)
    logger.debug(f"Latest model file detected: {latest_model}")
    return latest_model

# -----------------------------
# Register model
# -----------------------------
def register_model(model_name: str, model_file: str):
    try:
        model = joblib.load(model_file)

        with mlflow.start_run():  # 🔥 NEW: Let MLflow create run_id
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            logger.info(f"✅ Model '{model_name}' registered successfully")

        # Transition to Staging
        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=["None"])
        if latest_versions:
            version = latest_versions[0].version
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging"
            )
            logger.info(f"🚀 Model {model_name} v{version} moved to Staging")

    except Exception as e:
        logger.error(f"Error while registering the model: {e}")
        raise

# -----------------------------
# Main
# -----------------------------
def main():
    try:
        # Optional: load run info (but not using run_id anymore)
        model_info_path = "reports/model_info.json"
        if os.path.exists(model_info_path):
            _ = load_model_info(model_info_path)

        latest_model_file = get_latest_model_file()
        model_name = "my_model"
        register_model(model_name, latest_model_file)

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        raise

if __name__ == "__main__":
    main()
