# train.py
import argparse
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
import mlflow
from  mlflow.models import infer_signature
import mlflow.pytorch

# Import reusable components from the 'common' package
from common.model import DeckTransformer
from common.data import load_data
from common.config import load_and_validate_config

from common.evaluation import evaluate

COMMON_FILES_DIR="common"

def log_model_to_mlflow(model,input_example,epoch):
    signature = infer_signature(model_input=input_example.cpu().numpy(), 
                            model_output=model(input_example).detach().numpy())
    mlflow.pytorch.log_model(
        model, 
        name="deck_predictor_model",
        signature=signature,
        code_paths=[COMMON_FILES_DIR],
        step=epoch
    )

def train(args,config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup MLflow
    mlflow.set_tracking_uri(config['tracking_url'])
    mlflow.set_experiment(config['experiment_name'])
    
    train_loader, val_loader, _ = load_data(args.data_file, args.rows_to_load, args.batch_size)
    
    # Input example for mlflow
    input_example_tensor, _ = next(iter(train_loader))
    input_example = input_example_tensor.to(device)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting Training Run. MLflow Run ID: {run_id}")
        mlflow.set_tag("pipeline_stage", "training")
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
            if(args.log_model_every_epoch==True):
                log_model_to_mlflow(model,input_example,epoch)
        if(args.log_model_every_epoch!=True):
            log_model_to_mlflow(model,input_example,epoch) 
        
        print(f"\n--- Model trained successfully! Use this Run ID for evaluation: {run_id} ---")

if __name__ == '__main__':
    try:
        config = load_and_validate_config()
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train a Transformer model for deck prediction.")
    # MlfLow params
    parser.add_argument("--log_model_every_epoch", type=bool, default=False)
    # Data and model arguments
    parser.add_argument("--data_file", type=str, default="decks.csv")
    parser.add_argument("--rows_to_load", type=int, default=1000000)
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
    train(args,config)