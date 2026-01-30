from pydantic import BaseModel

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
# app/schemas.py
from pydantic import BaseModel


class PredictionInput(BaseModel):
    age: int
    bmi: float
    glucose_level: float
    blood_pressure: float
    insulin: float
    skin_thickness: float
    diabetes_pedigree_function: float


class OutputSchema(BaseModel):
    prediction: int
