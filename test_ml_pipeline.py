import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(os.path.exists("iris_dataset.csv"))

    def test_model_created(self):
        self.assertTrue(os.path.exists("iris_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)

    def test_model_prediction(self):
        model = joblib.load("iris_model.pkl")

        sample = pd.DataFrame([{
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }])

        prediction = model.predict(sample)[0]
        self.assertIn(int(prediction), [5])


if __name__ == "__main__":
    unittest.main()
