import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/outbreak_rf_model.pkl")

st.title("Disease Outbreak Predictor")
st.caption("Using AI to support SDG 3: Good Health and Well-being")

region = st.selectbox("Region", ["Region A", "Region B", "Region C"])
cases_last_week = st.number_input("Cases Last Week", min_value=0, value=120)
cases_2_weeks_ago = st.number_input("Cases Two Weeks Ago", min_value=0, value=100)
temperature = st.slider("Temperature (°C)", 0.0, 50.0, 30.0)
humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)

if st.button("Predict Outbreak Risk"):
    input_df = pd.DataFrame({
        "cases_last_week": [cases_last_week],
        "cases_2_weeks_ago": [cases_2_weeks_ago],
        "temperature": [temperature],
        "humidity": [humidity]
    })

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f" High Outbreak Risk: {prob*100:.1f}%")
    else:
        st.success(f" Low Outbreak Risk: {(1 - prob)*100:.1f}%")

    st.markdown("###  Input Summary")
    st.write(input_df)
