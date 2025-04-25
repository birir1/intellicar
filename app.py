import streamlit as st
import pandas as pd
from models.model_trainer import train_predictive_model  # Assuming you import this function

# Title for the app
st.title("CAN Data Threat Prediction")

# Sidebar for uploading files
st.sidebar.header("Upload Your Data")
can_file = st.sidebar.file_uploader("Upload CAN Data", type=["csv"])
threat_file = st.sidebar.file_uploader("Upload Threat Intel", type=["json"])

# Check if files are uploaded
if can_file and threat_file:
    # Load the CAN data and threat intel from the uploaded files
    can_data = pd.read_csv(can_file)
    nlp_features = pd.read_json(threat_file)

    # Show the uploaded data in the app
    st.subheader("Uploaded CAN Data")
    st.write(can_data.head())

    # Train the model and get predictions
    predictions, model, y_test, y_pred = train_predictive_model(can_data, nlp_features)

    # Display predictions
    st.subheader("Model Predictions")
    st.write(predictions[['speed', 'rpm', 'temp', 'brake', 'predicted_threat']])

    # Show model performance
    st.subheader("Model Performance")
    accuracy = (y_test == y_pred).mean()
    st.write(f"Accuracy: {accuracy:.2f}")
    
else:
    st.warning("Please upload both CAN Data and Threat Intel files.")
