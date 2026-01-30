import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

DATA_PATH = "data/data_cleaned_standardized_clustred_classified.csv"
TARGET = "Cluster"

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Cluster", "risk_category"])
y = df["Cluster"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric="logloss", random_state=42)
}

mlflow.set_experiment("Diabetes-Risk-Prediction")
client = MlflowClient()
best_acc = 0
best_model = None

for name, model_obj in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model_obj)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("model_type", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=name
        )

        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_model_name = name
            best_run_id = mlflow.active_run().info.run_id

if best_run_id:
    versions = client.get_latest_versions(best_model_name)
    for v in versions:
        if v.run_id == best_run_id:
            client.transition_model_version_stage(
                name=best_model_name,
                version=v.version,
                stage="Production",
                archive_existing_versions=True
            )

    os.makedirs("../models/production", exist_ok=True)
    mlflow.sklearn.save_model(
        sk_model=best_model,
        path="../models/production/best_model.joblib"
    )
    print(f"Model {best_model_name} promoted to Production")
