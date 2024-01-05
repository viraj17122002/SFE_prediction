import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# model_path = "path/to/your/joblib/model/file.joblib"



# Load the trained model
model_path = 'best_model.joblib'  # Replace with the actual path
# final_model_optimized = load_model(model_path)
final_model_optimized = joblib.load(model_path)

# Streamlit UI
st.title("SFE Prediction for Fe-Mn-Al-C-Si-Ni-Cr Alloy")

# Collect user inputs
feature_names = ['C', 'Ni', 'Al', 'Fe', 'Mn', 'Si', 'Cr']
user_input_features = []

# Initialize Fe content
fe_content = 100.0

for i in range(7):
    # # For all features, take user input
    # feature_value = st.number_input(f"Enter the % of {feature_names[i]}", value=0.0)
    if (feature_names[i]!="Fe"):
        feature_value = st.number_input(f"Enter the % of {feature_names[i]}", value=0.0)
        user_input_features.append(feature_value)
    else:   
        user_input_features.append(0.0)

# Calculate Fe content based on the entered values
fe_content -= sum(user_input_features)

# Update the Fe content in the user_input_features list
user_input_features[feature_names.index('Fe')] = fe_content

# Function to preprocess input and make a prediction
def predict_output(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Make a prediction using the user input
if st.button("Predict"):
    predicted_output = predict_output(final_model_optimized, user_input_features)
    st.success(f"Predicted Output: {predicted_output}")
