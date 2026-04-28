import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Smart Traffic Prediction", layout="wide")

st.title("🚦 Smart Traffic Congestion Prediction System")

# ================= SIDEBAR =================
st.sidebar.header("📥 Input Traffic Details")

hour = st.sidebar.slider("Hour", 0, 23)
day = st.sidebar.selectbox("Day", ["Weekday", "Weekend"])
weather = st.sidebar.selectbox("Weather", ["Clear", "Rain", "Fog"])
road = st.sidebar.selectbox("Road Condition", ["Good", "Moderate", "Bad"])
location = st.sidebar.text_input("Location (Optional)")

st.sidebar.metric("Model Accuracy", "92%")

# ================= SINGLE PREDICTION =================
input_data = pd.DataFrame({
    "hour": [hour],
    "day": [day],
    "weather": [weather],
    "road_condition": [road]
})

input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=columns, fill_value=0)
input_scaled = scaler.transform(input_data)

if st.button("🚀 Predict Traffic"):
    result = model.predict(input_scaled)[0]

    st.subheader("📊 Prediction Result")

    if result == "High":
        st.error("🚨 Heavy Traffic")
        st.info("💡 Avoid peak hours (8-10 AM, 5-8 PM)")
    elif result == "Medium":
        st.warning("⚠️ Moderate Traffic")
        st.info("💡 Plan travel with buffer time")
    else:
        st.success("✅ Low Traffic")
        st.info("💡 Best time to travel")

    # ================= SMALL GRAPH =================
    st.subheader("📈 Traffic Visualization")

    levels = ["Low", "Medium", "High"]
    values = [0, 0, 0]

    if result == "Low":
        values[0] = 1
    elif result == "Medium":
        values[1] = 1
    else:
        values[2] = 1

    # 👇 SMALL SIZE FIX
    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar(levels, values)
    ax.set_title("Traffic Level")

    # 👇 CENTER GRAPH
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.pyplot(fig, use_container_width=False)

# ================= BATCH PREDICTION =================
st.subheader("📂 Batch Prediction (Upload CSV)")

file = st.file_uploader("Upload CSV File", type=["csv"])

if file is not None:
    try:
        data = pd.read_csv(file)
        st.write("📊 Uploaded Data Preview", data.head())

        required_cols = ["hour", "day", "weather", "road_condition"]

        if not all(col in data.columns for col in required_cols):
            st.error("❌ CSV must contain: hour, day, weather, road_condition")
        else:
            data_encoded = pd.get_dummies(data)
            data_encoded = data_encoded.reindex(columns=columns, fill_value=0)

            data_scaled = scaler.transform(data_encoded)
            predictions = model.predict(data_scaled)

            data["Predicted Traffic"] = predictions

            st.success("✅ Batch Prediction Completed!")
            st.write(data)

            # 📊 SMALL CHART
            st.subheader("📊 Traffic Distribution")
            counts = data["Predicted Traffic"].value_counts()

            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.bar(counts.index, counts.values)

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.pyplot(fig2, use_container_width=False)

            # Download
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results", csv, "results.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# ================= FOOTER =================
st.markdown("---")
st.markdown("Developed using Ensemble ML + Streamlit 🚀")