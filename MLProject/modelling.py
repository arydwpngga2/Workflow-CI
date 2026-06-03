import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.sklearn.autolog()

df = pd.read_csv("telco_preprocessing.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

acc = accuracy_score(
    y_test,
    pred
)

print("Accuracy:", acc)

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/result.txt", "w") as f:
    f.write(f"Accuracy: {acc}")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print("RUN_ID =", run_id)

    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model,
        "model"
    )