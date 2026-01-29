import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score

MIN_ACCURACY = 0.9
MIN_F1_SCORE = 0.9

MODEL_URI = "models:/LogisticRegression/Production"  
model = mlflow.sklearn.load_model(MODEL_URI)

DATA_PATH = "data/diabetes_data_cleaned_classified.csv"
df = pd.read_csv(DATA_PATH)
X_test = df.drop(columns=["Cluster", "risk_category"])
y_test = df["Cluster"]

def test_model_accuracy():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    assert acc >= MIN_ACCURACY, f"Accuracy too low: {acc:.2f}"

def test_model_f1_score():
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    assert f1 >= MIN_F1_SCORE, f"F1-score too low: {f1:.2f}"
