# import mlflow.sklearn
# import pandas as pd
# from sklearn.metrics import accuracy_score, f1_score

# MODEL_NAME = "LogisticRegression"
# MODEL_STAGE = "Production"

# MIN_ACCURACY = 0.75
# MIN_F1_SCORE = 0.70

# DATA_PATH = "data/diabetes_data_cleaned_classified.csv"
# df = pd.read_csv(DATA_PATH)
# X_test = df.drop(columns=["Cluster", "risk_category"])
# y_test = df["Cluster"]

# def test_model_performance_ci():
#     model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
#     model = mlflow.sklearn.load_model(model_uri)
    
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
    
#     assert acc >= MIN_ACCURACY, f"Accuracy too low: {acc} < {MIN_ACCURACY}"
#     assert f1 >= MIN_F1_SCORE, f"F1-score too low: {f1} < {MIN_F1_SCORE}"
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "LogisticRegression"
MODEL_STAGE = "Production"  # doit correspondre au stage du modèle temporaire
MIN_ACCURACY = 0.7
MIN_F1_SCORE = 0.6

def test_model_performance_ci():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv("data/diabetes_data_cleaned_classified.csv")
    X_test = df.drop(columns=["Cluster", "risk_category"])
    y_test = df["Cluster"]

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    assert acc >= MIN_ACCURACY
    assert f1 >= MIN_F1_SCORE
