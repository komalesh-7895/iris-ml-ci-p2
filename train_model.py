import json
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model():
    iris = load_iris(as_frame=True)
    data = iris.frame

    data.to_csv("iris_dataset.csv", index=False)

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("Dataset records:", len(data))
    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))
    print("Accuracy:", round(accuracy, 4))
    print("Confusion Matrix:")
    print(matrix)
    print("Class names:", iris.target_names.tolist())

    joblib.dump(model, "iris_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "classes": iris.target_names.tolist()
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved successfully.")
    print("Metrics saved successfully.")


if __name__ == "__main__":
    train_model()
