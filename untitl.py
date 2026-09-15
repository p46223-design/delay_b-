import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Delivery Delay Prediction App",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Delivery Delay Prediction App")
st.write("Provide the shipment details below to predict whether a delivery will be delayed.")

# Load the trained model safely
@st.cache_resource
def load_model():
    return joblib.load('delivery_delay_model.sav')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file (`delivery_delay_model.sav`). Make sure it is uploaded to your repository. Details: {e}")
    st.stop()

# Create input form for the 11 features
with st.form("prediction_form"):
    st.subheader("Shipment & Route Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        delivery_distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=20.0, step=0.5)
        traffic_congestion = st.slider("Traffic Congestion (1-5)", min_value=1, max_value=5, value=3)
        weather_condition = st.selectbox("Weather Condition (1: Good, 2: Moderate, 3: Bad)", options=[1, 2, 3])
        delivery_slot = st.selectbox("Delivery Slot (1-3)", options=[1, 2, 3])
        driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=40, value=5)
        num_stops = st.number_input("Number of Stops", min_value=1, max_value=20, value=2)

    with col2:
        vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5)
        road_condition_score = st.slider("Road Condition Score (1-3)", min_value=1, max_value=3, value=2)
        package_weight = st.number_input("Package Weight (kg)", min_value=0.1, max_value=200.0, value=15.0, step=0.1)
        fuel_efficiency = st.number_input("Fuel Efficiency (km/L)", min_value=1.0, max_value=50.0, value=14.0, step=0.1)
        warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, max_value=300, value=45)

    submitted = st.form_submit_button("Predict Delivery Status")

# Handle prediction output
if submitted:
    # Arrange input data into the exact feature order expected by the model
    input_data = pd.DataFrame([[
        delivery_distance,
        traffic_congestion,
        weather_condition,
        delivery_slot,
        driver_experience,
        num_stops,
        vehicle_age,
        road_condition_score,
        package_weight,
        fuel_efficiency,
        warehouse_processing_time
    ]], columns=[
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 
        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 
        'Warehouse_Processing_Time'
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"⚠️ **Prediction: Delayed Delivery** (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.success(f"✅ **Prediction: On-Time Delivery** (Confidence: {prediction_proba[0]*100:.2f}%)")