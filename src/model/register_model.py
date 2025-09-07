import os
import json
import logging
import mlflow
from mlflow import MlflowClient

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - model_registration - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model_registration")

# ----------------------------
# Load model info (from pipeline)
# ----------------------------
def load_model_info(path="reports/model_info.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model info file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

# ----------------------------
# Register model safely
# ----------------------------
def register_and_alias_model(model_name, model_uri, alias="staging"):
    """
    Register a model in MLflow and assign alias (like staging/production).
    """
    client = MlflowClient()
    status = "unknown"

    try:
        logger.info(f"🔄 Attempting to register model '{model_name}' at {model_uri}")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        status = "registered"
        logger.info(f"✅ Model '{model_name}' registered successfully.")

        # Fetch latest version
        latest_version = client.get_latest_versions(model_name, stages=[])[-1].version
        logger.info(f"ℹ️ Latest version for {model_name} is v{latest_version}")

        # Assign alias
        client.set_registered_model_alias(model_name, alias, latest_version)
        logger.info(f"🏷️ Alias '{alias}' assigned to version {latest_version}")

    except Exception as e:
        logger.warning(f"⚠️ Model registry may not be supported: {e}")
        status = "logged_only"

    # Save registration info (so DVC has a file output)
    os.makedirs("reports", exist_ok=True)
    out_path = "reports/registered_model.json"
    with open(out_path, "w") as f:
        json.dump(
            {"model_name": model_name, "model_uri": model_uri, "status": status},
            f,
            indent=4,
        )

    logger.info(f"📄 Registration info written to {out_path}")
    return status


# ----------------------------
# Main entry
# ----------------------------
def main():
    # Setup MLflow with DagsHub tracking server
    mlflow.set_tracking_uri("https://dagshub.com/Vaibha3246/mlops-mini-project.mlflow")
    logger.info(f"Accessing as {os.getenv('DAGSHUB_USERNAME', 'unknown_user')}")

    # Load model info produced by earlier pipeline stage
    model_info = load_model_info()
    model_name = model_info.get("model_name", "my_model")
    model_uri = model_info.get("model_uri")

    logger.info(f"ℹ️ Model '{model_name}' available at {model_uri}")

    # Register and assign alias
    register_and_alias_model(model_name, model_uri, alias="staging")


if __name__ == "__main__":
    main()
