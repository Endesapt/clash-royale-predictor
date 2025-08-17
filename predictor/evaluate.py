# evaluate.py
import argparse
import sys
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException, RestException
from tqdm import tqdm
import torch.nn as nn

from common.data import load_data
from common.config import load_and_validate_config
from common.evaluation import evaluate

def evaluate_and_promote(args, config):
    """
    Evaluates a 'challenger' model, compares it against the current 'champion' model,
    and promotes the challenger by assigning it the 'champion' alias if its performance is better.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Setup MLflow
    mlflow.set_tracking_uri(config['tracking_url'])
    mlflow.set_experiment(config['experiment_name'])
    client = MlflowClient()

    # Determine if a higher score is better for the chosen metric
    higher_is_better = "loss" not in args.sort_metric.lower()

    # --- Start a new MLflow run for this evaluation and promotion process ---
    with mlflow.start_run() as run:
        print(f"Starting Evaluation Run. Results will be logged to Run ID: {run.info.run_id}")
        mlflow.set_tag("pipeline_stage", "evaluation_and_promotion")
        mlflow.log_params(vars(args))

        # 2. Evaluate the CHALLENGER model
        try:
            print(f"Fetching challenger model with alias '{args.challenger_alias}'...")
            challenger_version = client.get_model_version_by_alias(
                name=args.registered_model_name, alias=args.challenger_alias
            )
            print(f"Found challenger: Version {challenger_version.version} from Run ID {challenger_version.run_id}")
            mlflow.set_tag("challenger_run_id", challenger_version.run_id)
            mlflow.set_tag("challenger_model_version", challenger_version.version)

            challenger_model = mlflow.pytorch.load_model(challenger_version.source, map_location=device)
        except (RestException, MlflowException) as e:
            print(f"Error: Could not retrieve challenger model with alias '{args.challenger_alias}'. "
                  f"Please ensure a model version has this alias. Error: {e}", file=sys.stderr)
            sys.exit(1)

        _, _, test_loader = load_data(args.data_file, args.rows_to_load, args.batch_size)
        criterion = nn.CrossEntropyLoss(ignore_index=args.padding_idx)

        # Run evaluation logic
        _, challenger_accuracies = evaluate(challenger_model, test_loader, criterion, device)
        challenger_score = challenger_accuracies.get(args.sort_metric)

        if challenger_score is None:
            print(f"Error: Metric '{args.sort_metric}' not found in evaluation results for challenger.", file=sys.stderr)
            sys.exit(1)

        print(f"Challenger Model Score ({args.sort_metric}): {challenger_score:.4f}")
        mlflow.log_metric(f"challenger_{args.sort_metric}", challenger_score)
        
        # It's good practice to also log the score back to the original training run
        client.log_metric(challenger_version.run_id, args.sort_metric, challenger_score)


        # 3. Get the CHAMPION (current baseline) model's performance
        champion_score = -1.0 if higher_is_better else float('inf')
        champion_version = None

        try:
            champion_version = client.get_model_version_by_alias(
                name=args.registered_model_name, alias=args.champion_alias
            )
            champion_run = client.get_run(champion_version.run_id)
            champion_score = champion_run.data.metrics.get(args.sort_metric)
            
            if champion_score is None:
                 print(f"Warning: Metric '{args.sort_metric}' not found in champion run '{champion_version.run_id}'. "
                       "Will proceed assuming challenger is better.", file=sys.stderr)
                 # Reset score to default to ensure promotion
                 champion_score = -1.0 if higher_is_better else float('inf')
            else:
                print(f"Found champion model: Version {champion_version.version} from Run ID {champion_version.run_id}")
                print(f"Champion Model Score ({args.sort_metric}): {champion_score:.4f}")
                mlflow.log_metric(f"champion_{args.sort_metric}", champion_score)
                mlflow.set_tag("champion_model_version", champion_version.version)

        except (RestException, MlflowException):
            print(f"No champion model found with alias '{args.champion_alias}'. This challenger will be promoted by default.")

        # 4. Compare models and promote if challenger is better
        promotion_decision = False
        if higher_is_better:
            if challenger_score > champion_score:
                promotion_decision = True
        else: # Lower is better (e.g., loss)
            if challenger_score < champion_score:
                promotion_decision = True
        
        mlflow.log_param("promotion_decision", promotion_decision)

        if promotion_decision:
            print(f"Challenger model is better. Promoting to '{args.champion_alias}'.")
            
            # Set the champion alias to the challenger version. MLflow handles moving the alias.
            print(f"Setting alias '{args.champion_alias}' to version {challenger_version.version}...")
            client.set_registered_model_alias(
                name=args.registered_model_name,
                alias=args.champion_alias,
                version=challenger_version.version
            )

            # Clean up by removing the challenger alias from the now-promoted model
            print(f"Removing '{args.challenger_alias}' alias...")
            client.delete_registered_model_alias(
                name=args.registered_model_name,
                alias=args.challenger_alias
            )

            print("Promotion successful.")
            mlflow.set_tag("promoted_model_version", challenger_version.version)

        else:
            print("Challenger model is not better than the champion. No promotion.")

if __name__ == '__main__':
    try:
        config = load_and_validate_config()
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
        
    parser = argparse.ArgumentParser(description="Evaluate a 'challenger' model, compare to a 'champion', and promote if better.")
    parser.add_argument("--registered_model_name", type=str, required=True, help="Name of the model in the MLflow Model Registry.")
    parser.add_argument("--challenger_alias", type=str, default="challenger", help="The alias of the candidate model to evaluate.")
    parser.add_argument("--champion_alias", type=str, default="champion", help="The alias of the baseline model to compare against.")
    parser.add_argument(
        "--sort_metric", 
        type=str, 
        default="top10_accuracy", 
        help="The metric to use for comparing the models."
    )
    # Inherited arguments
    parser.add_argument("--padding_idx", type=int, default=120)
    parser.add_argument("--data_file", type=str, default="decks.csv")
    parser.add_argument("--rows_to_load", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()
    evaluate_and_promote(args, config)