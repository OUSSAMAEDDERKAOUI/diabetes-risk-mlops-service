import pandas as pd
import sys

EXPECTED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

df = pd.read_csv("data/data_cleaned_standardized_clustred_classified.csv")

missing = set(EXPECTED_COLUMNS) - set(df.columns)
if missing:
    print(f"Missing columns: {missing}")
    sys.exit(1)

if (df["Glucose"] <= 0).any():
    print("Invalid glucose values")
    sys.exit(1)

print("Data validation passed ")
