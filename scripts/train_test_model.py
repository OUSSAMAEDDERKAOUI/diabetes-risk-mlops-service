import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys

MIN_ACCURACY = 0.75
MIN_F1_SCORE = 0.70

MODEL_URI = "models:/LogisticRegression/Production"  
model = mlflow.sklearn.load_model(MODEL_URI)

DATA_PATH = "data/diabetes_data_cleaned_classified.csv"
df = pd.read_csv(DATA_PATH)

X_test = df.drop(columns=["Cluster", "risk_category"])
y_test = df["Cluster"]

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

if acc < MIN_ACCURACY or f1 < MIN_F1_SCORE:
    print(" Model performance below threshold!")
    sys.exit(1)  
else:
    print(" Model performance OK!")
    sys.exit(0)  
