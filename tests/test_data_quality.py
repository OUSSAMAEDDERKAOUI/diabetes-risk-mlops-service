import pandas as pd
import glob
import os
import pytest

DATA_DIR = "data/"
EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Cluster",
    "risk_category"
]
RISK_CATEGORIES = ["risque_faible", "risque_moyen", "risque_eleve"]


csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
if not csv_files:
    pytest.fail("Aucun fichier CSV trouvé dans le dossier data/")


@pytest.mark.parametrize("file_path", csv_files)
def test_data_quality(file_path):
    df = pd.read_csv(file_path)

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    assert not missing_columns, f"Colonnes manquantes dans {file_path}: {missing_columns}"

    for col in ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Colonne {col} doit être numérique dans {file_path}"
        assert df[col].notna().all(), f"Colonne {col} contient des valeurs nulles dans {file_path}"

@pytest.mark.parametrize("file_path", csv_files)
def test_no_missing_values(file_path):
    df = pd.read_csv(file_path)
    assert df.isnull().sum().sum() == 0