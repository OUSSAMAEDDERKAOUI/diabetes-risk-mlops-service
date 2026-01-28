FROM python:3.10-slim
WORKDIR /mlflow
COPY . /mlflow
RUN pip install --no-cache-dir -r requirements/requirements-ml.txt
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--default-artifact-root", "file:/mlflow/artifacts"]
