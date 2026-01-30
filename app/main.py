# from fastapi import FastAPI
# from app.schemas import PatientData
# from app.model_loader import load_model
# import pandas as pd

# app = FastAPI()
# model = load_model()

# @app.post("/predict")
# def predict(patient: PatientData):
#     df = pd.DataFrame([patient.dict()])
#     risk = model.predict(df)[0]
#     return {"risk_category": "risque_eleve" if risk==1 else "risque_faible"}

# @app.get("/health")
# def health():
#     return {"status": "API is running"}
# from prometheus_client import Counter, Histogram, generate_latest
# from fastapi import Response

# REQUEST_COUNT = Counter("request_count", "Total API requests")
# REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")

# @app.get("/metrics")
# def metrics():
#     return Response(generate_latest(), media_type="text/plain")
# app/main.py
from fastapi import FastAPI
from app.model_loader import load_model
from app.schemas import PredictionInput
import pandas as pd

app = FastAPI(title="Diabetes Risk API")

model = load_model()

@app.post("/predict")
def predict(data: PredictionInput):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)
    return {"prediction": int(pred[0])}
