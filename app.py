import streamlit as st
import pandas as pd
import joblib
import json
from streamlit_lottie import st_lottie

# =================================================
# PAGE CONFIG
# =================================================
st.set_page_config(
    page_title="AERISENSE | Air Quality Intelligence",
    layout="wide"
)

# =================================================
# GLOBAL STYLING (BACKGROUND + SIDEBAR)
# =================================================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #6EC1E4, #8E7CC3);
}

/* Sidebar title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 22px !important;
    font-weight: 700;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] label {
    font-size: 18px !important;
    font-weight: 600;
    color: #ffffff !important;
}

/* Main text */
h1, h2, h3 {
    color: #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# HELPER FUNCTION
# =================================================
def load_lottie(path):
    with open(path, "r") as f:
        return json.load(f)

# =================================================
# LOAD LOTTIE (ONLY SECTION LOTTIES)
# =================================================
lottie_predict = load_lottie("flow_predict.json")
lottie_city = load_lottie("flow_city.json")

# =================================================
# LOAD DATA & MODEL
# =================================================
df = pd.read_csv("city_day.csv")
df = df.dropna(subset=["AQI"])
df["Date"] = pd.to_datetime(df["Date"])

model = joblib.load("aqi_model.pkl")

# =================================================
# AQI CATEGORY FUNCTION
# =================================================
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

# =================================================
# HERO / HEADER (TEXT ONLY – CLEAN)
# =================================================
st.markdown("""
<div style="text-align:center; padding: 40px 0 20px 0;">
    <h1 style="
        font-size:72px;
        font-weight:800;
        background: linear-gradient(90deg, #6EC1E4, #8E7CC3);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 10px;
    ">
        AERISENSE
    </h1>
    <p style="font-size:20px; color:#cbd5f5;">
        Air Quality & Environmental Intelligence Platform
    </p>
</div>
<hr>
""", unsafe_allow_html=True)

# =================================================
# SIDEBAR NAVIGATION
# =================================================
st.sidebar.markdown("## 🔧 Navigation")

section = st.sidebar.radio(
    "Go to",
    ["🔮 AQI Prediction", "🏙️ City-wise AQI Analysis"]
)

# =================================================
# SECTION 1: ML AQI PREDICTION
# =================================================
pm25 = pm10 = no2 = so2 = co = o3 = 0.0

if section == "🔮 AQI Prediction":
    st.header("🔮 ML-Based AQI Prediction")

    colL, colR = st.columns([1, 2])

    with colL:
        st_lottie(lottie_predict, height=240, speed=1)

    with colR:
        c1, c2, c3 = st.columns(3)

        with c1:
            pm25 = st.number_input("PM2.5", min_value=0.0)
            pm10 = st.number_input("PM10", min_value=0.0)

        with c2:
            no2 = st.number_input("NO₂", min_value=0.0)
            so2 = st.number_input("SO₂", min_value=0.0)

        with c3:
            co = st.number_input("CO", min_value=0.0)
            o3 = st.number_input("O₃", min_value=0.0)

        if st.button("Predict AQI"):
            features = [[pm25, pm10, no2, so2, co, o3]]
            prediction = model.predict(features)

            predicted_aqi = int(prediction[0])
            category = aqi_category(predicted_aqi)

            st.success(f"Predicted AQI: {predicted_aqi}")
            st.info(f"AQI Category: {category}")

# =================================================
# SECTION 2: CITY-WISE AQI ANALYSIS
# =================================================
if section == "🏙️ City-wise AQI Analysis":
    st.header("🏙️ City-wise AQI Analysis")

    st_lottie(lottie_city, height=220, speed=1)

    city = st.selectbox("Select City", sorted(df["City"].unique()))
    city_data = df[df["City"] == city]

    min_date = city_data["Date"].min()
    max_date = city_data["Date"].max()

    start_date, end_date = st.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    filtered_data = city_data[
        (city_data["Date"] >= pd.to_datetime(start_date)) &
        (city_data["Date"] <= pd.to_datetime(end_date))
    ]

    avg_aqi = int(filtered_data["AQI"].mean())
    category = aqi_category(avg_aqi)

    colA, colB = st.columns(2)
    colA.metric("Average AQI", avg_aqi)
    colB.metric("AQI Category", category)

    st.line_chart(filtered_data.set_index("Date")["AQI"])
