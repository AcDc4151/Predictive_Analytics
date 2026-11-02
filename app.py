
# ==========================================
# PART C2: Interactive Prototype Development
# Cooling Load Prediction App
# ==========================================

import joblib
import pandas as pd
import streamlit as st

# ------------------------------------------
# Step 1: Load the trained model
# ------------------------------------------

MODEL_PATH = "Best_Cooling_Load_Model.joblib"

st.set_page_config(
    page_title="Cooling Load Predictor",
    page_icon="🏢",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Please place "
        "Best_Cooling_Load_Model.joblib in the same folder as app.py."
    )
    st.stop()

# ------------------------------------------
# Step 2: App title and description
# ------------------------------------------

st.title("🏢 Building Cooling Load Predictor")

st.write(
    "Adjust the building design parameters below "
    "to predict the Cooling Load (kWh/m²)."
)

st.info(
    "Select a building profile, adjust its parameters, "
    "and click Predict Cooling Load."
)

# ------------------------------------------
# Step 3: Input validation ranges
# ------------------------------------------

feature_bounds = {
    "X1": (0.5, 1.0),
    "X2": (514.5, 808.5),
    "X3": (245.0, 416.5),
    "X4": (110.25, 220.5),
    "X5": (3.5, 7.0),
    "X6": (2, 5),
    "X7": (0.0, 0.5),
    "X8": (0, 5)
}

feature_labels = {
    "X1": "Relative Compactness",
    "X2": "Surface Area",
    "X3": "Wall Area",
    "X4": "Roof Area",
    "X5": "Overall Height",
    "X6": "Orientation",
    "X7": "Glazing Area",
    "X8": "Glazing Distribution"
}

orientation_labels = {
    2: "North",
    3: "East",
    4: "South",
    5: "West"
}

# ------------------------------------------
# Step 4: Choose a building profile
# ------------------------------------------

profile = st.selectbox(
    "Select Building Profile",
    ["Design A", "Design B", "Design C", "Custom Design"]
)

# Editable starting values.
# Replace these with the exact values from the assignment table.
default_profiles = {
    "Design A": {
        "X1": 0.85, "X2": 588.0, "X3": 294.0,
        "X4": 147.0, "X5": 7.0, "X6": 2,
        "X7": 0.25, "X8": 3
    },
    "Design B": {
        "X1": 0.65, "X2": 784.0, "X3": 343.0,
        "X4": 220.5, "X5": 3.5, "X6": 4,
        "X7": 0.10, "X8": 1
    },
    "Design C": {
        "X1": 0.90, "X2": 563.5, "X3": 318.5,
        "X4": 122.5, "X5": 7.0, "X6": 3,
        "X7": 0.40, "X8": 4
    },
    "Custom Design": {
        "X1": 0.75, "X2": 650.0, "X3": 300.0,
        "X4": 150.0, "X5": 3.5, "X6": 2,
        "X7": 0.10, "X8": 2
    }
}

defaults = default_profiles[profile]

# ------------------------------------------
# Step 5: Interactive input controls
# ------------------------------------------

st.subheader("Building Design Parameters")

col1, col2 = st.columns(2)

input_values = {}

with st.form("cooling_load_form"):

    for i, feature in enumerate(feature_bounds):
        low, high = feature_bounds[feature]
        label = f"{feature} - {feature_labels[feature]}"

        container = col1 if i < 4 else col2

        with container:
            if feature == "X6":
                input_values[feature] = st.selectbox(
                    label,
                    options=[2, 3, 4, 5],
                    index=[2, 3, 4, 5].index(
                        int(defaults[feature])
                    ),
                    format_func=lambda x: (
                        f"{x} ({orientation_labels[x]})"
                    )
                )

            elif feature == "X8":
                input_values[feature] = st.slider(
                    label,
                    min_value=int(low),
                    max_value=int(high),
                    value=int(defaults[feature]),
                    step=1
                )

            elif feature == "X5":
                input_values[feature] = st.select_slider(
                    label,
                    options=[3.5, 7.0],
                    value=float(defaults[feature])
                )

            else:
                step = 0.01 if feature in ["X1", "X7"] else 0.5

                input_values[feature] = st.number_input(
                    label,
                    min_value=float(low),
                    max_value=float(high),
                    value=float(defaults[feature]),
                    step=float(step)
                )

    predict_button = st.form_submit_button(
        "Predict Cooling Load",
        type="primary",
        use_container_width=True
    )

# ------------------------------------------
# Step 6: Predict and classify energy grade
# ------------------------------------------

def classify_cooling_load(y_pred):
    if y_pred <= 15.0:
        return "A", "Excellent", "#006400"
    elif y_pred <= 20.0:
        return "B", "Good", "#90EE90"
    elif y_pred <= 28.0:
        return "C", "Average", "#FFD700"
    elif y_pred <= 35.0:
        return "D", "Below Average", "#FFA500"
    else:
        return "E", "Poor", "#DC2626"


if predict_button:

    # Keep the same feature order used during training
    input_df = pd.DataFrame(
        [[input_values[f"X{i}"] for i in range(1, 9)]],
        columns=[f"X{i}" for i in range(1, 9)]
    )

    # Validate the inputs
    valid = True

    for feature, (low, high) in feature_bounds.items():
        value = input_values[feature]

        if not low <= value <= high:
            st.error(
                f"{feature} must be between {low} and {high}."
            )
            valid = False

    if valid:
        # Run prediction
        prediction = float(model.predict(input_df)[0])

        grade, description, colour = classify_cooling_load(
            prediction
        )

        st.divider()
        st.subheader("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Predicted Cooling Load",
                f"{prediction:.2f} kWh/m²"
            )

        with result_col2:
            st.markdown(
                f"""
                <div style="
                    background-color: {colour};
                    padding: 18px;
                    border-radius: 12px;
                    text-align: center;
                    color: black;
                ">
                    <h2>Energy Grade: {grade}</h2>
                    <h3>{description}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Visual gauge
        st.subheader("Cooling Load Gauge")

        gauge_value = max(0.0, min(prediction, 35.0))

        st.progress(gauge_value / 35.0)

        st.caption(
            "Lower predicted Cooling Load indicates lower "
            "cooling energy demand. The gauge is capped at 35 "
            "for visualisation; predictions above 35 are "
            "classified as Poor."
        )

        st.subheader("Input Summary")
        st.dataframe(input_df, use_container_width=True)

# ------------------------------------------
# Step 7: Display grade reference
# ------------------------------------------

st.divider()
st.subheader("Energy Grade Reference")

grade_reference = pd.DataFrame({
    "Grade": ["A", "B", "C", "D", "E"],
    "Cooling Load (kWh/m²)": [
        "Y <= 15",
        "15 < Y <= 20",
        "20 < Y <= 28",
        "28 < Y <= 35",
        "Y > 35"
    ],
    "Performance": [
        "Excellent",
        "Good",
        "Average",
        "Below Average",
        "Poor"
    ]
})

st.table(grade_reference)

st.caption(
    "Predictions are estimates from the trained machine "
    "learning model and should be interpreted as model outputs."
)
