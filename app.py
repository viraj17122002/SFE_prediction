import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path = 'model.h5'  # Replace with the actual path
final_model_optimized = load_model(model_path)

# Streamlit UI
st.title("Model Prediction App")

# Collect user inputs
feature_names=['C', 'N', 'P', 'S', 'V', 'Ni', 'Nb', 'Al', 'Ti', 'Fe', 'Hf', 'Mo', 'Mn',
       'Co', 'Si', 'Cr', 'Cu', 'temperature']
user_input_features = []
for i in range(18):
    feature_value = st.number_input(f"Enter the % of {feature_names[i]}", value=0.0)
    user_input_features.append(feature_value)

# Function to preprocess input and make a prediction
def predict_output(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0][0]/10

# Make a prediction using the user input
if st.button("Predict"):
    predicted_output = predict_output(final_model_optimized, user_input_features)
    st.success(f"Predicted Output: {predicted_output}")
