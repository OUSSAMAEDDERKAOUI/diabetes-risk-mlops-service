import mlflow
import mlflow.sklearn

MODEL_NAME = "LogisticRegression"

def load_model():
    model_uri = f"models:/{MODEL_NAME}/Production"  
    model = mlflow.sklearn.load_model(model_uri)
    return model
