# import mlflow
# import mlflow.sklearn

# MODEL_NAME = "LogisticRegression"

# def load_model():
#     model_uri = f"models:/{MODEL_NAME}/Production"  
#     model = mlflow.sklearn.load_model(model_uri)
#     return model

# app/model_loader.py
import os
import joblib

MODEL_PATH = os.path.join("models", "production", "best_model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


