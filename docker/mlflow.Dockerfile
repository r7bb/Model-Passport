# MLflow tracking server with the S3 client needed to use MinIO as the artifact store.
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install "mlflow>=3,<4" boto3 \
 && useradd --create-home --uid 10001 mlflow && mkdir -p /mlflow && chown mlflow /mlflow
USER mlflow
EXPOSE 5000
