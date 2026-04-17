import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()

# ------------------ CUSTOM STYLE ------------------
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🤖 Machine Learning Prediction App")
st.write("Enter the input values below to get predictions.")

st.divider()

# ------------------ INPUT FIELDS ------------------
# ⚠️ CHANGE THESE BASED ON YOUR MODEL FEATURES
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)
    feature2 = st.number_input("Feature 2", value=0.0)

with col2:
    feature3 = st.number_input("Feature 3", value=0.0)
    feature4 = st.number_input("Feature 4", value=0.0)

# ------------------ PREDICTION ------------------
if st.button("🔍 Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3, feature4]])
        prediction = model.predict(input_data)

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ------------------ FOOTER ------------------
st.divider()
st.caption("Built with ❤️ using Streamlit")
