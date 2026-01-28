import pandas as pd
# import mlflow
# import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


DATA_PATH = "./data/data_cleaned_standardized_clustred_classified.csv"
TARGET = "risk_category"
MODEL_NAME = "diabetes-risk-model"


df = pd.read_csv(DATA_PATH)
print(df.columns)
X = df.drop(columns=[TARGET])
y = df[TARGET]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    ))
])

