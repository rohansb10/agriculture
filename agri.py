import streamlit as st
import joblib
import numpy as np

# Load your trained model
model_file_path = r'C:\Users\Rohan\Pictures\rohan\BigMart_Sales project\agri_project\model_file.pkl'
loaded_model = joblib.load(model_file_path)

# Create a Streamlit app
st.title("Agriculture Prediction App")

# Set background color
st.markdown(
    """
    <style>
        body {
            background-color: #f0f0f0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Get user input
nitrogen = st.slider("Select Nitrogen level:", min_value=0, max_value=1000, step=1)
phosphorus = st.slider("Select Phosphorus level:", min_value=0, max_value=300, step=1)
potassium = st.slider("Select Potassium level:", min_value=0, max_value=300, step=1)
temperature = st.slider("Select Temperature level:", min_value=0, max_value=100, step=1)
humidity = st.slider("Select Humidity level:", min_value=0.0, max_value=100.0, step=0.1)
ph = st.slider("Select pH level:", min_value=0.0, max_value=14.0, step=0.1)
rainfall = st.slider("Select Rainfall level:", min_value=0.0, max_value=200.0, step=0.1)

# Make prediction
if st.button("Get Prediction"):
    input_data = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
    input_array = np.array(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    st.success(f"The predicted result is: {prediction[0]}")
