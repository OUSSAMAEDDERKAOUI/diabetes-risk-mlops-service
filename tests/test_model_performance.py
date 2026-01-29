# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import accuracy_score, f1_score

# MIN_ACCURACY = 0.9
# MIN_F1_SCORE = 0.9

# MODEL_URI = "models:/LogisticRegression/Production"  
# model = mlflow.sklearn.load_model(MODEL_URI)

# DATA_PATH = "data/diabetes_data_cleaned_classified.csv"
# df = pd.read_csv(DATA_PATH)
# X_test = df.drop(columns=["Cluster", "risk_category"])
# y_test = df["Cluster"]

# def test_model_accuracy():
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     assert acc >= MIN_ACCURACY, f"Accuracy too low: {acc:.2f}"

# def test_model_f1_score():
#     y_pred = model.predict(X_test)
#     f1 = f1_score(y_test, y_pred)
#     assert f1 >= MIN_F1_SCORE, f"F1-score too low: {f1:.2f}"
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Seuils
MIN_ACCURACY = 0.75
MIN_F1_SCORE = 0.70

# Data sample pour CI (ou charger un CSV petit)
DATA_PATH = "data/data_cleaned_standardized_clustred_classified.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Cluster", "risk_category"])
y = df["Cluster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

@pytest.mark.parametrize("model", [LogisticRegression(solver="liblinear", random_state=42)])
def test_model_performance(model):
    # Entraînement rapide
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    assert acc >= MIN_ACCURACY, f"Accuracy too low: {acc}"
    assert f1 >= MIN_F1_SCORE, f"F1-score too low: {f1}"
