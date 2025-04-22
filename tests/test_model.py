import os
import joblib
import numpy as np
from sklearn.datasets import load_iris

def test_model_file_exists():
    assert os.path.exists("model.pkl"), "model.pkl file was not found. Did you run train.py?"

def test_model_can_predict():
    model = joblib.load("model.pkl")
    iris = load_iris()
    sample = iris.data[0].reshape(1, -1)
    prediction = model.predict(sample)
    assert prediction[0] in [0, 1, 2], "Invalid prediction output"
