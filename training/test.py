from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "LogisticRegression"  

versions = client.get_latest_versions(name=model_name, stages=["Production"])

if versions:
    prod_version = versions[0]
    print(f"Modèle {model_name} en Production : version {prod_version.version}")
    print(f"Run ID : {prod_version.run_id}")
    print(f"URI du modèle : {prod_version.source}")
else:
    print(f"Aucune version de {model_name} en Production.")
