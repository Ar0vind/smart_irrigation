import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("irrigation_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Smart Irrigation System", layout="wide")
st.title("AI-Powered Smart Irrigation System")
st.markdown("### Enter Raw Sensor Readings (0 - 1023)")
sensor_values = []
for i in range(20):
    val = st.number_input(f"Sensor {i}", min_value=0, max_value=1023, value=500, step=1)
    sensor_values.append(val)

def predict_irrigation(sensor_input):
    scaled_input = scaler.transform([sensor_input])
    prediction = model.predict(scaled_input)[0]
    probs = model.predict_proba(scaled_input)
    return prediction, probs

if st.button("Predict Irrigation Status"):
    pred, prob = predict_irrigation(sensor_values)

    st.subheader("Prediction Result")
    st.subheader("Prediction Result")
    for i, (status, p) in enumerate(zip(pred, prob)):

        try:
            prob_on = float(p[0][1])  
        except:
         try:
            prob_on = float(p[1])  
         except:
            prob_on = 0.0      

    percent = int(prob_on * 100)
    color = "green" if status == 1 else "red"
    st.markdown(
        f"<span style='color:{color}; font-size:18px;'>Parcel {i} Sprinkler: {'ON' if status else 'OFF'} ({percent}% confidence)</span>",
        unsafe_allow_html=True
    )


    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((sensor_values, pred.tolist()))
if st.checkbox("Show Prediction History"):
    if "history" in st.session_state:
        hist_df = pd.DataFrame([
            {"Sensor " + str(i): val for i, val in enumerate(item[0])} |
            {"Parcel " + str(i): p for i, p in enumerate(item[1])}
            for item in st.session_state.history
        ])
        st.dataframe(hist_df)


if st.checkbox("Show Feature Importances"):
    importances = np.mean([
        tree.feature_importances_ for tree in model.estimators_[0].estimators_
    ], axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(20), importances)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Sensor Index")
    ax.set_ylabel("Importance")
    st.pyplot(fig)
