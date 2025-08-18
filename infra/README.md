# Installation guides

## IMPORTANT!
Before installation of any chart, look at values.yaml for your chart if it has something like `existingSecret: minio-credentials` and create this secrets in your space with content required


## Install CertManager and apply ClusterIssuer
```bash 
cd certs
helm install \
   cert-manager oci://quay.io/jetstack/charts/cert-manager \
   --version v1.18.2 \
   --namespace cert-manager \
   --create-namespace \
   --set crds.enabled=true \
   -f values.yaml
kubectl apply -f cluster-issuer.yaml  
```
## Install Kserve for model serving

- Install [CertManager](https://cert-manager.io/)
- Install IngressController or gateway (we used nginx)
- Install helm chart from kserve folder using:
    ```bash 
    helm upgrade --install kserve oci://ghcr.io/kserve/charts/kserve --version v0.15.0 -f values.yaml -n kserve --create-namespace
    ```

## Install Airflow
From airflow folder:
```bash 
    helm repo add apache-airflow https://airflow.apache.org
    helm upgrade --install  --version 1.18.0 airflow apache-airflow/airflow --namespace airflow --version 1.16.0 --create-namespace -f values.yaml
```

## Install Minio
From minio folder:
```bash 
    helm upgrade --install --version 17.0.19 my-release oci://registry-1.docker.io/bitnamicharts/minio -n minio --create-namespace -f values.yaml
```

## Install Mlflow
From airflow folder:
```bash 
    helm repo add community-charts https://community-charts.github.io/helm-charts
    helm upgrade --install  mlflow community-charts/mlflow -n mlflow --create-namespace -f values.yaml
```
