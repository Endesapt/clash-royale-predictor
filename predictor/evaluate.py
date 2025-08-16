# evaluate.py
import argparse
import sys
import torch
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.entities import LoggedModelInput
from tqdm import tqdm
import torch.nn as nn

from common.data import load_data
from common.config import load_and_validate_config
from common.evaluation import evaluate

def evaliate_model(args,config):
    """
    Loads a trained model and evaluates it on the test set using mlflow.evaluate().
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1. Setup MLflow
    mlflow.set_tracking_uri(config['tracking_url'])
    mlflow.set_experiment(config['experiment_name'])

    print(f"Searching for the best model in run '{args.model_run_id}' sorted by '{args.sort_metric}'...")

    sort_ascending = "loss" in args.sort_metric.lower()
    
    try:
        ranked_models = mlflow.search_logged_models(
            filter_string=f"source_run_id='{args.model_run_id}'",
            order_by=[{"field_name": f"metrics.{args.sort_metric}", "ascending": (True if sort_ascending else False)} ],
            output_format="list"
        )
    except mlflow.exceptions.MlflowException as e:
        print(f"Error searching for models: {e}", file=sys.stderr)
        print("Please ensure the training run logged models and the specified metric exists.", file=sys.stderr)
        sys.exit(1)
    
    if not ranked_models:
        print(f"Error: No models found in run '{args.model_run_id}'.", file=sys.stderr)
        sys.exit(1)
    
    best_model_info = ranked_models[0]
    model_uri = best_model_info.model_uri
    print(f"Found best model. URI: {model_uri}")

    # 2. Load Test Data.
    _, _, test_loader = load_data(args.data_file, args.rows_to_load,args.batch_size)
    criterion = nn.CrossEntropyLoss(ignore_index=args.padding_idx)
    

    with mlflow.start_run() as run:
        print(f"Starting Evaluation Run for model: {model_uri}")
        mlflow.set_tag("pipeline_stage", "evaluation")
        mlflow.set_tag("evaluated_model_uri", model_uri)
        mlflow.log_param("sort_metric_used", args.sort_metric)
        mlflow.log_input(model=LoggedModelInput(best_model_info.model_id))

        # Load Model
        model = mlflow.pytorch.load_model(model_uri, map_location=device)
        # Call the shared evaluation function on the test set
        avg_loss, accuracies = evaluate(model, test_loader, criterion, device)
        
        mlflow.log_metrics(accuracies)
            
        print(f"\nView full results in the MLflow UI for Run ID: {run.info.run_id}")



if __name__ == '__main__':
    try:
        config = load_and_validate_config()
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model using mlflow.evaluate().")
    parser.add_argument("--model_run_id", type=str, required=True, help="MLflow Run ID of the trained model to evaluate.")
    parser.add_argument(
        "--sort_metric", 
        type=str, 
        default="top10_accuracy", 
        choices=["loss", "top1_accuracy", "top5_accuracy", "top10_accuracy"],
        help="The metric to use for finding the best model."
    )
    parser.add_argument("--padding_idx", type=int, default=120)
    parser.add_argument("--data_file", type=str, default="decks.csv")
    parser.add_argument("--rows_to_load", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128) # Can use a larger batch for faster data loading

    args = parser.parse_args()
    evaliate_model(args,config)