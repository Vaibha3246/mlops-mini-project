# register_model.py

import json
import mlflow
import logging
import os
import glob
import dotenv
from dotenv import load_dotenv

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
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

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
    """Load model info from JSON."""
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
    """Return the latest .pkl model file from src/model/ or project root."""
    search_dirs = ["src/model", "."]  # first look in src/model, then root
    pkl_files = []

    for d in search_dirs:
        files = glob.glob(os.path.join(d, "*.pkl"))
        pkl_files.extend(files)

    if not pkl_files:
        raise FileNotFoundError(
            f"No .pkl model files found in {', '.join(search_dirs)}"
        )

    latest_model = max(pkl_files, key=os.path.getctime)
    logger.debug(f"Latest model file detected: {latest_model}")
    return latest_model

# -----------------------------
# Register / Log model
# -----------------------------
def register_model(model_name: str, model_info: dict, model_file: str):
    """Log the latest model file to DagsHub MLflow."""
    try:
        # Start MLflow run using run_id from model_info
        with mlflow.start_run(run_id=model_info["run_id"]):
            mlflow.log_artifact(model_file, artifact_path=model_name)
            logger.info(f"Model '{model_name}' logged successfully from '{model_file}'")
            print(f"✅ Model '{model_name}' logged successfully from '{model_file}'")
    except Exception as e:
        logger.error(f"Error while logging the model: {e}")
        raise

# -----------------------------
# Main function
# -----------------------------
def main():
    try:
        # Load model info
        model_info_path = "reports/model_info.json"
        model_info = load_model_info(model_info_path)

        # Automatically detect latest model
        latest_model_file = get_latest_model_file()

        # Register the model
        model_name = "my_model"
        register_model(model_name, model_info, latest_model_file)

    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        print(f"Error: {e}")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main()
