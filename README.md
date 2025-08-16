# Clash Royal Predictor



## Testing Airflow DAGS connections:
Add include/connections.yaml file with this contents:
```yaml
clash_royale_api:
  conn_type: http
  conn_id: clash_royale_api
  host: https://api.clashroyale.com
  extra:
    Authorization: "Bearer <your token>"

minio_s3:
  conn_type: aws
  conn_id: minio_s3
  login: <your login>
  password: <your_pass>
  extra:
    endpoint_url: <endpoint_url>
```

start with:
```
pip install -f dags/requirements.txt
python dags/royal-api-etl.py
```