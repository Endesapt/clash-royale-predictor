# Building docker image
```bash
docker build . -t trainer
```

# Training and Evaluating
Before starting training or evaluating u should provide .env file (see .env.example for reference) or provide this runtime variables
```conf
MLFLOW_S3_ENDPOINT_URL=https://your-s3-provider.com
AWS_ACCESS_KEY_ID=YOUR_KEY_HERE
AWS_SECRET_ACCESS_KEY=YOUR_SECRET_HERE

MLFLOW_TRACKING_URI=http://127.0.0.1:5000
MLFLOW_EXPERIMENT_NAME=MyClashExperiment
```

To start training
```bash
pip install -r requirements.txt

# it is recommended to set --rows_to_load to be equal for train and evaluation

# Train
python train.py --epochs=5 --rows_to_load=1000000 --registerModel 

# Evaluate
python train.py --rows_to_load=1000000 
```