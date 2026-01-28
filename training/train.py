import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = "data/data_cleaned_standardized_clustred_classified.csv"  
TARGET = "risk_category"
MODEL_NAME = "diabetes-risk-model"


df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Cluster","risk_category"])
y = df["Cluster"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric="logloss", random_state=42)
}


mlflow.set_experiment("Diabetes-Risk-Prediction-MLflow")
client=MlflowClient()
best_acc=0
best_run_id = None
for model_name, model_obj in models.items():

    with mlflow.start_run(run_name=model_name):

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model_obj)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", model_name)
        if model_name == "LogisticRegression":
            mlflow.log_param("solver", "liblinear")
        elif model_name == "RandomForest":
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 8)
        elif model_name == "XGBoost":
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 6)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=model_name
        )

        print(f" {model_name} trained, logged and registered in MLflow")
        
        if acc>best_acc:
            best_acc=acc
            best_run_id=mlflow.active_run().info.run_id
            print(f"{model_name} trained, logged and registered in MLflow")

if best_run_id:
    versions=client.get_latest_versions(name=model_name)
    for v in versions:
        if v.run_id==best_run_id:
            client.transition_model_version_stage(name=model_name,version=v.version,stage="Production",archive_existing_versions=True)
        print(f"Model {model_name} version {v.version} promoted to Production")
