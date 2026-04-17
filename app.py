import streamlit as st
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="ML Prediction App",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for styling
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
            font-size: 16px;
        }
        .stTextInput>div>div>input {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Machine Learning Prediction App")
st.write("Enter input values below to get predictions from your trained model.")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Input section (modify based on your model features)
st.subheader("📥 Input Features")

# Example inputs (CHANGE based on your model)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Prediction button
if st.button("🔍 Predict"):
    try:
        input_data = np.array([[feature1, feature2, feature3]])
        prediction = model.predict(input_data)

        st.success(f"✅ Prediction Result: {prediction[0]}")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
