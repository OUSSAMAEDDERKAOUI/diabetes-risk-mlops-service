import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_PATH = "data/diabetes.csv"  
TARGET = "risk_category"
MODEL_NAME = "diabetes-risk-model"


df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET])
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


models = {
    "LogisticRegression": LogisticRegression(solver="liblinear", random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, use_label_encoder=False, eval_metric="logloss", random_state=42)
}


mlflow.set_experiment("Diabetes-Risk-Prediction")

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

       
        