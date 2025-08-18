## Testing Airflow DAGS connections:

## IMPORTANT!
before testing u need to initialize airflow local sqlite database using
```bash
airflow db migrate
```

Add include/connections.yaml (from repo root) file with this contents:
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

start test with:
```bash
pip install -f dags/requirements.txt
python dags/royal-api-etl.py
```

To change dag that u want to test change this code part in dags/royal-api-etl.py
```python
test_dag=api_to_minio_enrichment_dag()

if __name__ == "__main__":
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    with open(os.path.join(parent_directory,"..","include","connections.yaml")) as file:
        print(file.name)
    #chage to test_dag if you want to test ETL pipeline instead of training pipeline
    dag.test(
        conn_file_path=os.path.join(parent_directory,"..","include","connections.yaml"),
        run_conf={"rows_to_load":10000}
    )
```

