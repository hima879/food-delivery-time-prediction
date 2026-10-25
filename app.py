"""
app.py
------
Streamlit dashboard for the Food Delivery Time Prediction System.
Run from project root:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

from src.utils import (
    load_model, validate_inputs, preprocess_single_row,
    predict_delivery_time, error_margin,
)

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
)

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Could not load model. Run `python src/model_train.py`. Error: {e}")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Dashboard"])
st.sidebar.markdown("---")
if MODEL_LOADED:
    st.sidebar.info(f"Deployed Model: **{type(model.model).__name__}**")
    st.sidebar.info(f"R² Score: **{model.metrics.get('r2', 0):.3f}**")


if page == "🔮 Predict":
    st.title("🛵 Predict Delivery Time")
    st.write("Enter the order details below to estimate delivery time.")

    if not MODEL_LOADED:
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Route Details")
            distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
            city = st.selectbox("City Type", ["Urban", "Metropolitian", "Semi-Urban"])
            order_hour = st.slider("Order Hour (24h)", 0, 23, 19)

        with col2:
            st.subheader("Conditions")
            weather = st.selectbox("Weather", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"])
            traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
            festival = st.selectbox("Festival?", ["No", "Yes"])
            order_type = st.selectbox("Order Type", ["Snack", "Meal", "Drinks", "Buffet"])
            vehicle_type = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])

        with col3:
            st.subheader("Delivery Person")
            age = st.number_input("Age", 15, 65, 30)
            ratings = st.number_input("Rating (0-5)", 0.0, 5.0, 4.5, step=0.1)
            vehicle_condition = st.selectbox("Vehicle Condition (0-3)", [0, 1, 2, 3], index=2)
            multiple_deliveries = st.selectbox("Multiple Deliveries", [0, 1, 2, 3])

        submit = st.form_submit_button("Predict Delivery Time", use_container_width=True)

    if submit:
        is_valid, error_msg = validate_inputs(
            distance_km, age, ratings, order_hour, multiple_deliveries, vehicle_condition
        )
        if not is_valid:
            st.error(error_msg)
        else:
            input_df = preprocess_single_row(
                distance_km=distance_km, weather=weather, traffic=traffic,
                order_hour=order_hour, city=city, order_type=order_type,
                vehicle_type=vehicle_type, multiple_deliveries=multiple_deliveries,
                festival=festival, age=age, ratings=ratings,
                vehicle_condition=vehicle_condition,
            )
            pred_time = predict_delivery_time(model, input_df)
            margin = error_margin(model)

            st.success(f"### Estimated Delivery Time: **{pred_time} minutes**")
            st.write(f"*(Expected error margin: ± {margin:.1f} minutes)*")

            input_df["Predicted_Time_min"] = pred_time
            csv = input_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Prediction Data (CSV)",
                data=csv,
                file_name="delivery_prediction.csv",
                mime="text/csv",
            )

            st.markdown("### Simulated Route")
            base_lat, base_lon = 12.9716, 77.5946
            m = folium.Map(location=[base_lat, base_lon], zoom_start=12)
            folium.Marker([base_lat, base_lon], popup="Restaurant").add_to(m)
            folium.Marker(
                [base_lat + distance_km * 0.001, base_lon + distance_km * 0.001],
                popup="Delivery Location",
            ).add_to(m)
            st_folium(m, width=700, height=400, returned_objects=[])


elif page == "📊 Dashboard":
    st.title("📊 Model Performance Dashboard")

    if not MODEL_LOADED:
        st.stop()

    try:
        df = pd.read_csv("data/processed/food_delivery_processed.csv")
    except Exception:
        df = None

    st.subheader("Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{model.metrics['r2']:.3f}")
    col2.metric("RMSE", f"{model.metrics['rmse']:.2f} min")
    col3.metric("MAE", f"{model.metrics['mae']:.2f} min")
    col4.metric("MSE", f"{model.metrics['mse']:.2f}")

    st.markdown("---")
    st.subheader("Baseline vs Deployed Models")
    st.table(pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "HistGradientBoosting"],
        "R²":    [0.577, 0.820, 0.822],
        "RMSE":  [5.892, 3.845, 3.820],
        "MAE":   [4.709, 3.058, 3.044],
    }))

    st.markdown("---")
    if df is not None:
        st.subheader("Processed Dataset Preview")
        st.write(f"Rows: **{df.shape[0]:,}** | Columns: **{df.shape[1]}**")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("EDA & Results Plots")

    plot_files = sorted(
        [f for f in os.listdir("assets/plots") if f.endswith(".png")]
    ) if os.path.isdir("assets/plots") else []

    if not plot_files:
        st.warning("No plots found. Run `python src/visualize.py` first.")
    else:
        for i in range(0, len(plot_files), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(plot_files):
                    img_path = os.path.join("assets/plots", plot_files[i + j])
                    with cols[j]:
                        st.image(
                            img_path,
                            caption=plot_files[i + j].replace(".png", "").replace("_", " ").title(),
                            use_container_width=True,
                        )
