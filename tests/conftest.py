import os
import pytest
import pandas as pd

# ------------------
# Paths
# ------------------
DATA_PATH = "data/diabetes_data_cleaned_classified.csv"
CANDIDATE_DIR = "models/candidate"

# ------------------
# Thresholds
# ------------------
MIN_ACC = 0.75
MIN_F1 = 0.70

# ------------------
# Fixtures pytest
# ------------------
@pytest.fixture(scope="session")
def dataset():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Cluster", "risk_category"])
    y = df["Cluster"]
    return X, y

@pytest.fixture(scope="session")
def candidate_models():
    models = []
    for f in os.listdir(CANDIDATE_DIR):
        if f.endswith(".joblib"):
            models.append(os.path.join(CANDIDATE_DIR, f))

    assert len(models) > 0, "❌ No candidate models found"
    return models
