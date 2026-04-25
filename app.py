import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("🚦 Traffic Prediction")

hour = st.slider("Hour", 0, 23)
day = st.selectbox("Day", ["Weekday", "Weekend"])
weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"])
road = st.selectbox("Road Condition", ["Good", "Moderate", "Bad"])

input_data = pd.DataFrame({
    "hour": [hour],
    "day": [day],
    "weather": [weather],
    "road_condition": [road]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=columns, fill_value=0)

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    result = model.predict(input_scaled)
    st.success(f"🚗 Traffic: {result[0]}")