# lets use streamlit to create a simple app
import streamlit as st
import numpy as np
import pandas as pd
import pickle
# load the model
with open('logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Iris Flower Species Prediction")
st.write("Enter the features of the iris flower to predict its species.")

# create input fields for the features
sepal_length = st.number_input("Sepal Length", min_value=0.0)
sepal_width = st.number_input("Sepal Width", min_value=0.0)
petal_length = st.number_input("Petal Length", min_value=0.0)
petal_width = st.number_input("Petal Width", min_value=0.0)

# make prediction
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    species = ["Setosa", "Versicolor", "Virginica"]
    st.write(f"The predicted species is: {species[prediction[0]]}")
    