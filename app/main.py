from fastapi import FastAPI
from app.schemas import PatientData
from app.model_loader import load_model
import pandas as pd

app = FastAPI()
model = load_model()

@app.post("/predict")
def predict(patient: PatientData):
    df = pd.DataFrame([patient.dict()])
    risk = model.predict(df)[0]
    return {"risk_category": "risque_eleve" if risk==1 else "risque_faible"}

@app.get("/health")
def health():
    return {"status": "API is running"}
