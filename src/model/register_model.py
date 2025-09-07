import os
import json
import logging
import mlflow
from dagshub import dagshub_logger

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - model_registration - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_registration")

# ----------------------------
# Load model info
# ----------------------------
def load_model_info(path="reports/model_info.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model info file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

# ----------------------------
# Safe model registration
# ----------------------------
def safe_register_model(model_name, model_uri):
    """
    Try to register the model. 
    If registry is not supported (like in DagsHub), fall back gracefully.
    """
    try:
        logger.info(f"🔄 Attempting to register model '{model_name}' at {model_uri}")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        status = "registered"
        logger.info(f"✅ Model '{model_name}' registered successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Model registry not supported here: {e}")
        status = "logged_only"
    
    # Always write output file so DVC is happy
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/registered_model.json"
    with open(out_path, "w") as f:
        json.dump({"model_name": model_name, "model_uri": model_uri, "status": status}, f, indent=4)
    
    logger.info(f"📄 Registration info written to {out_path}")
    return status

# ----------------------------
# Main entry
# ----------------------------
def main():
    # Setup MLflow with DagsHub
    logger.info(f"Accessing as {os.getenv('DAGSHUB_USERNAME', 'unknown_user')}")
    mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/mlops-mini-project.mlflow")

    # Load model info (produced in earlier pipeline stages)
    model_info = load_model_info()
    model_name = model_info.get("model_name", "my_model")
    model_uri = model_info.get("model_uri")

    logger.info(f"ℹ️ Model '{model_name}' is available at: {model_uri}")

    # Try safe registration
    safe_register_model(model_name, model_uri)


if __name__ == "__main__":
    main()
