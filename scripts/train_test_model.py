import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/diabetes_data_cleaned_classified.csv"
MODEL_NAME = "LogisticRegression"
MODEL_STAGE = "Production"  # le stage que le test CI va charger

# Charger les données
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Cluster", "risk_category"])
y = df["Cluster"]

# Split rapide pour CI
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline simple
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(solver="liblinear"))
])

# Entraînement
pipeline.fit(X_train, y_train)

# Logger et enregistrer dans MLflow
with mlflow.start_run(run_name=MODEL_NAME):
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        registered_model_name=MODEL_NAME
    )
    mlflow.log_param("solver", "liblinear")
    print("Temporary CI model trained and registered.")
