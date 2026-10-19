
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢"
)

st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

try:
    model = load_model()

    feature_names = getattr(model, "feature_names_in_", None)
    number_of_features = getattr(model, "n_features_in_", None)

    if feature_names is not None:
        st.subheader("Passenger Features")
        values = {}

        for feature in feature_names:
            values[feature] = st.number_input(
                str(feature),
                value=0.0,
                step=1.0
            )

        input_data = pd.DataFrame([values])

    elif number_of_features is not None:
        st.warning(
            "Model feature names are unavailable. "
            "Enter the encoded numeric features in the "
            "same order used during training."
        )

        values = []
        for i in range(number_of_features):
            values.append(
                st.number_input(
                    f"Feature {i + 1}",
                    value=0.0,
                    step=1.0
                )
            )

        input_data = pd.DataFrame([values])

    else:
        st.error("Could not determine the model's input features.")
        st.stop()

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction")
        if str(prediction) == "1":
            st.success("The model predicts: Survived")
        elif str(prediction) == "0":
            st.error("The model predicts: Did not survive")
        else:
            st.info(f"Model output: {prediction}")

except FileNotFoundError:
    st.error(
        "Model file not found. Upload "
        "svc_trained_model.pkl to the same GitHub repository."
    )
except Exception as error:
    st.error(f"Prediction error: {error}")
