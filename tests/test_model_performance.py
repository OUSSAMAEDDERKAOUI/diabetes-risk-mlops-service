# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from sklearn.metrics import accuracy_score, f1_score

# MIN_ACCURACY = 0.75
# MIN_F1_SCORE = 0.70

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
import joblib
from sklearn.metrics import accuracy_score, f1_score
from conftest import MIN_ACC, MIN_F1


def test_candidate_models_performance(dataset, candidate_models):
    X, y = dataset

    for model_path in candidate_models:
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print(f"{model_path} → ACC={acc:.3f}, F1={f1:.3f}")

        assert acc >= MIN_ACC, f" {model_path} ACC too low"
        assert f1 >= MIN_F1, f" {model_path} F1 too low"
