from pathlib import Path
import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "model" / "student_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🎓 Student Performance Prediction")
st.markdown(
    "Predict a student's **final grade (G3)** using study habits and previous academic performance."
)

st.divider()

# -----------------------------
# User Input
# -----------------------------
studytime = st.number_input(
    "📚 Study Time (Hours per Week)",
    min_value=0.0,
    max_value=30.0,
    value=5.0,
    step=1.0,
)

failures = st.number_input(
    "❌ Previous Class Failures",
    min_value=0,
    max_value=4,
    value=0,
)

absences = st.number_input(
    "📅 Number of Absences",
    min_value=0,
    max_value=100,
    value=0,
)

g1 = st.number_input(
    "📝 G1 Grade (First Period)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
)

g2 = st.number_input(
    "📝 G2 Grade (Second Period)",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Final Grade", use_container_width=True):

    features = np.array(
        [[studytime, failures, absences, g1, g2]]
    )

    prediction = model.predict(features)[0]

    st.success(f"🎯 Predicted Final Grade (G3): **{prediction:.2f} / 20**")

    if prediction >= 16:
        st.balloons()
        st.success("🌟 Excellent Performance!")

    elif prediction >= 10:
        st.info("👍 Good Performance. Keep it up!")

    else:
        st.warning("📚 Needs Improvement. Focus on study time and attendance.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Developed using Streamlit and Scikit-learn")