# train.py
import argparse
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
from mlflow.models import infer_signature
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Import reusable components from the 'common' package
from common.model import DeckTransformer
from common.data import load_data
from common.config import load_and_validate_config
from common.evaluation import evaluate

COMMON_FILES_DIR="common"

def train(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['tracking_url'])
    mlflow.set_experiment(config['experiment_name'])
    client = MlflowClient()
    
    train_loader, val_loader, _ = load_data(args.data_file, args.rows_to_load, args.batch_size)
    
    # Input example for mlflow signature
    input_example_tensor, _ = next(iter(train_loader))
    input_example = input_example_tensor.to(device)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting Training Run. MLflow Run ID: {run_id}")
        # --- Add tags for better filtering ---
        mlflow.set_tag("pipeline_stage", "training")
        mlflow.set_tag("run_source", args.run_source) # <-- NEW
        
        mlflow.log_params(vars(args))

        model = DeckTransformer(
            vocab_size=args.vocab_size, embedding_dim=args.embedding_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, dropout=args.dropout, padding_idx=args.padding_idx
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=args.padding_idx)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        for epoch in range(args.epochs):
            model.train()
            total_train_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss, accuracies = evaluate(model, val_loader, criterion, device)
            
            metrics_to_log = {
                "train_loss": avg_train_loss,
                "eval_loss": avg_val_loss,
                "top1_accuracy": accuracies["top1_accuracy"],
                "top3_accuracy": accuracies["top3_accuracy"],
                "top5_accuracy": accuracies["top5_accuracy"],
                "top10_accuracy": accuracies["top10_accuracy"],
            }
            print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val top_10 Acc: {accuracies['top10_accuracy']*100:.2f}%")
            mlflow.log_metrics(metrics_to_log, step=epoch)
            
        # --- Final Model Logging and Registration ---
        print("\nTraining finished. Logging final model artifact...")
        artifact_path = "deck_predictor_model"
        signature = infer_signature(
            model_input=input_example.cpu().numpy(), 
            model_output=model(input_example).detach().cpu().numpy()
        )
        
        # 1. Always log the model as an artifact within the run
        mlflow.pytorch.log_model(
            model, 
            artifact_path=artifact_path,
            signature=signature,
            code_paths=[COMMON_FILES_DIR]
        )
        print(f"Model artifact logged to run {run_id} at path '{artifact_path}'")

        # 2. Conditionally register the model and apply an alias
        if args.register_model:
            if not args.registered_model_name:
                print("Error: --registered_model_name must be provided when --register_model is True.", file=sys.stderr)
                sys.exit(1)
            
            print(f"Registering model to '{args.registered_model_name}'...")
            model_uri = f"runs:/{run_id}/{artifact_path}"
            
            registered_version = mlflow.register_model(
                model_uri=model_uri,
                name=args.registered_model_name
            )
            print(f"Registered new version: {registered_version.version}")

            if args.model_alias:
                print(f"Setting alias '{args.model_alias}' for version {registered_version.version}...")
                client.set_registered_model_alias(
                    name=args.registered_model_name,
                    alias=args.model_alias,
                    version=registered_version.version
                )
                print("Alias set successfully.")
        
        print(f"\n--- Model trained successfully! Use this Run ID for evaluation: {run_id} ---")

if __name__ == '__main__':
    try:
        config = load_and_validate_config()
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train a Transformer model for deck prediction.")
    
    # MLflow Registration and Aliasing Arguments
    parser.add_argument("--register_model", action='store_true', help="If set, register the trained model in the MLflow Model Registry.")
    parser.add_argument("--registered_model_name", type=str, default=None, help="Name of the model in the registry. Required if --register_model is set.")
    parser.add_argument("--model_alias", type=str, default="challenger", help="Alias to apply to the registered model version (e.g., 'challenger').")
    parser.add_argument("--run_source", type=str, default="manual", help="Source of the run execution (e.g., 'airflow', 'manual'). Used for tagging.") 

    # Data and model arguments
    parser.add_argument("--data_file", type=str, default="decks.csv")
    parser.add_argument("--rows_to_load", type=int, default=10000)
    parser.add_argument("--vocab_size", type=int, default=121)
    parser.add_argument("--padding_idx", type=int, default=120)

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.15)
    
    args = parser.parse_args()
    train(args, config)