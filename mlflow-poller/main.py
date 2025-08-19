import os
import json
import requests
import time
import threading
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from flask import Flask

# --- Flask App for Health Checks ---
app = Flask(__name__)

@app.route("/health")
def health_check():
    return "OK", 200

# --- Environment Variables ---
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
POLLING_INTERVAL = int(os.environ.get("POLLING_INTERVAL", 60))
TARGET_ALIASES = ["champion", "production"]

# --- NEW: Optional specific model name to track ---
TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL_NAME") # If not set, this will be None

# --- In-Memory State ---
PROCESSED_EVENTS = set()

def trigger_github_action(model_name, version, alias, model_uri):
    """Constructs the payload and sends a dispatch request to GitHub Actions."""
    print(f"Triggering GitHub Action for {model_name} v{version} with alias '{alias}'")

    github_payload = {
        "event_type": "deploy-model-event",
        "client_payload": {
            "model_name": model_name,
            "model_version": version,
            "alias": alias,
            "model_uri": model_uri,
        }
    }

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }
    url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"

    try:
        response = requests.post(url, headers=headers, json=github_payload)
        response.raise_for_status()
        print(f"Successfully triggered GitHub Action for {model_name} v{version}.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error triggering GitHub Action: {e}")
        print(f"Response: {e.response.text if e.response else 'No response'}")
        return False

def poll_mlflow_for_aliases():
    """The main polling loop that checks MLflow for new model aliases."""
    print("--- Starting MLflow Polling Service ---")
    if TARGET_MODEL_NAME:
        print(f"Mode: Tracking specific model -> '{TARGET_MODEL_NAME}'")
    else:
        print("Mode: Tracking ALL registered models.")
    print(f"Polling Interval: {POLLING_INTERVAL} seconds")
    print(f"Target Aliases: {TARGET_ALIASES}")

    while True:
        try:
            mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            models_to_check = []

            # --- MODIFIED: Logic to select which models to poll ---
            if TARGET_MODEL_NAME:
                print(f"Polling for specific model: '{TARGET_MODEL_NAME}'")
                try:
                    # Get the specific registered model and put it in a list to iterate
                    model_obj = mlflow_client.get_registered_model(name=TARGET_MODEL_NAME)
                    models_to_check.append(model_obj)
                except MlflowException as e:
                    if "RESOURCE_DOES_NOT_EXIST" in str(e):
                        print(f"Warning: Target model '{TARGET_MODEL_NAME}' not found. Will try again next cycle.")
                    else:
                        raise e # Re-raise other MLflow exceptions
            else:
                print("Polling for all registered models...")
                # Get all registered models
                models_to_check = mlflow_client.search_registered_models()

            for model in models_to_check:
                for alias in TARGET_ALIASES:
                    try:
                        model_version_details = mlflow_client.get_model_version_by_alias(
                            name=model.name,
                            alias=alias
                        )
                        event_key = f"{model.name}:{model_version_details.version}:{alias}"

                        if event_key not in PROCESSED_EVENTS:
                            print(f"New event detected: Model '{model.name}' version '{model_version_details.version}' now has alias '{alias}'.")
                            success = trigger_github_action(
                                model_name=model.name,
                                version=model_version_details.version,
                                alias=alias,
                                model_uri=model_version_details.source
                            )
                            if success:
                                PROCESSED_EVENTS.add(event_key)
                        
                    except MlflowException as e:
                        if "RESOURCE_DOES_NOT_EXIST" in str(e):
                            pass 
                        else:
                            print(f"An MLflow error occurred for model '{model.name}': {e}")
            
            print(f"Polling cycle complete. Waiting {POLLING_INTERVAL} seconds.")

        except Exception as e:
            print(f"An unexpected error occurred in the polling loop: {e}")
        
        time.sleep(POLLING_INTERVAL)


if __name__ == "__main__":
    if not all([GITHUB_REPO, GITHUB_TOKEN, MLFLOW_TRACKING_URI]):
        raise ValueError("Missing one or more required environment variables: GITHUB_REPO, GITHUB_TOKEN, MLFLOW_TRACKING_URI")

    polling_thread = threading.Thread(target=poll_mlflow_for_aliases, daemon=True)
    polling_thread.start()
    app.run(host='0.0.0.0', port=8080)