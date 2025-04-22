import streamlit as st
import joblib
import numpy as np

st.title("🌸 Iris Species Predictor")

sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    model = joblib.load("model.pkl")
    prediction = model.predict(features)
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Predicted Iris species: {species[prediction[0]]}")
