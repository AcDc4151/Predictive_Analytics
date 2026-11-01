import streamlit as st

st.title("Cooling Load Prediction System")

st.write("Welcome to the Cooling Load Prediction System!")

import streamlit as st
import joblib
import numpy as np

# Load the trained Baseline MLR model
loaded_model = joblib.load("baseline_mlr_model.joblib")

st.write("Enter the building characteristics below to predict Cooling Load.")

# Input fields
st.header("Building Characteristics")

X1 = st.number_input("Relative Compactness (X1)", min_value=0.0, max_value=1.0, value=0.5)
X2 = st.number_input("Surface Area (X2)", min_value=0.0, value=600.0)
X3 = st.number_input("Wall Area (X3)", min_value=0.0, value=300.0)
X4 = st.number_input("Roof Area (X4)", min_value=0.0, value=150.0)
X5 = st.number_input("Overall Height (X5)", min_value=0.0, value=7.0)
X6 = st.number_input("Orientation (X6)", min_value=2, max_value=5, value=3)
X7 = st.number_input("Glazing Area (X7)", min_value=0.0, max_value=1.0, value=0.1)
X8 = st.number_input("Glazing Area Distribution (X8)", min_value=0, max_value=5, value=2)

# Prediction Button
if st.button("Predict Cooling Load"):

    # Arrange input values
    X_input = np.array([[X1, X2, X3, X4, X5, X6, X7, X8]])

    # Make prediction
    prediction = loaded_model.predict(X_input)

    # Display prediction result
    st.subheader("Prediction Result")

    st.metric(
        label="Predicted Cooling Load",
        value=f"{prediction[0]:.2f}"
    )

    st.success("Prediction completed successfully!")

