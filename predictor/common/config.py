# common/config.py
import os
from dotenv import load_dotenv

def load_and_validate_config():
    """
    Loads environment variables from a .env file (if it exists),
    validates that the required S3 variables are set, and returns them.
    
    Raises:
        ValueError: If any required environment variable is not set.
    """
    # Load variables from .env file. This is great for local development.
    # It will not override existing environment variables.
    load_dotenv()

    # Get configuration from environment variables
    config = {
        'mlflow_s3_endpoint_url': os.getenv('MLFLOW_S3_ENDPOINT_URL'),
        'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'tracking_url': os.getenv('MLFLOW_TRACKING_URI', "file:./mlruns"), # Default to local if not set
        'experiment_name': os.getenv('MLFLOW_EXPERIMENT_NAME', "ClashRoyale") # Default if not set
    }

    # Validate that required S3 variables are set
    required_s3_vars = [
        'mlflow_s3_endpoint_url', 
        'aws_access_key_id', 
        'aws_secret_access_key'
    ]
    
    missing_vars = [key for key, value in config.items() if key in required_s3_vars and not value]

    if missing_vars:
        error_message = (
            f"Error: Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please set them in your environment or create a '.env' file in the project root with the following content:\n\n"
            "MLFLOW_S3_ENDPOINT_URL=your_s3_endpoint_url\n"
            "AWS_ACCESS_KEY_ID=your_access_key\n"
            "AWS_SECRET_ACCESS_KEY=your_secret_key\n"
        )
        raise ValueError(error_message)
    
    # Set the environment variables for the current process, which MLflow will use
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = config['mlflow_s3_endpoint_url']
    os.environ['AWS_ACCESS_KEY_ID'] = config['aws_access_key_id']
    os.environ['AWS_SECRET_ACCESS_KEY'] = config['aws_secret_access_key']

    return config