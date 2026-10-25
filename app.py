"""
app.py
------
Streamlit dashboard for the Food Delivery Time Prediction System.
Run from project root:
    streamlit run app.py
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

from src.utils import (
    load_model, validate_inputs, preprocess_single_row,
    predict_delivery_time, error_margin,
)

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
)

# --------------------------------------------------------------------------
# GLOBAL STYLE
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #FF512F 0%, #F09819 50%, #FF512F 100%);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    border-radius: 18px;
    padding: 2.2rem 2rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(240, 152, 25, 0.35);
}
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero h1 {
    color: white;
    font-weight: 800;
    font-size: 2.1rem;
    margin-bottom: 0.3rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.15);
}
.hero p {
    color: rgba(255,255,255,0.92);
    font-size: 1.02rem;
    margin: 0;
}

/* Section subheaders */
.section-tag {
    display: inline-block;
    background: rgba(240, 152, 25, 0.12);
    color: #F09819;
    font-weight: 700;
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    margin-bottom: 0.6rem;
}

/* Prediction result card */
.result-card {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    color: white;
    box-shadow: 0 8px 24px rgba(17, 153, 142, 0.3);
}
.result-card h2 {
    margin: 0;
    font-size: 2.6rem;
    font-weight: 800;
}
.result-card p {
    margin: 0.2rem 0 0 0;
    opacity: 0.92;
}

/* Sidebar model badge */
.model-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: rgba(240, 152, 25, 0.06);
    border: 1px solid rgba(240, 152, 25, 0.18);
    border-radius: 14px;
    padding: 0.8rem 0.6rem;
}

/* Buttons */
.stButton>button, .stFormSubmitButton>button {
    background: linear-gradient(135deg, #FF512F 0%, #F09819 100%);
    color: white;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, .stFormSubmitButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(240, 152, 25, 0.4);
    color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"Could not load model. Run `python src/model_train.py`. Error: {e}")

# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.markdown("## 🛵 Delivery Predictor")
page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 Dashboard"], label_visibility="collapsed")
st.sidebar.markdown("---")

if MODEL_LOADED:
    st.sidebar.markdown(
        f"""
        <div class="model-badge">
            <div style="font-size:0.75rem; opacity:0.7; text-transform:uppercase; letter-spacing:0.05em;">Deployed Model</div>
            <div style="font-size:1.1rem; font-weight:700;">🏆 {type(model.model).__name__}</div>
        </div>
        <div class="model-badge">
            <div style="font-size:0.75rem; opacity:0.7; text-transform:uppercase; letter-spacing:0.05em;">R² Score</div>
            <div style="font-size:1.1rem; font-weight:700;">{model.metrics.get('r2', 0):.3f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")
st.sidebar.caption("BCA V Semester · Data Analytics using Python (BCA503)")

WEATHER_EMOJI = {
    "Sunny": "☀️", "Stormy": "⛈️", "Sandstorms": "🌪️",
    "Cloudy": "☁️", "Fog": "🌫️", "Windy": "💨",
}
TRAFFIC_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Jam": "🔴"}

# --------------------------------------------------------------------------
# PREDICT PAGE
# --------------------------------------------------------------------------
if page == "🔮 Predict":
    st.markdown("""
    <div class="hero">
        <h1>🛵 Predict Delivery Time</h1>
        <p>Fill in the order details and get an instant, data-driven delivery estimate.</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<span class="section-tag">Route</span>', unsafe_allow_html=True)
            distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)
            city = st.selectbox("City Type", ["Urban", "Metropolitian", "Semi-Urban"])
            order_hour = st.slider("Order Hour (24h)", 0, 23, 19)

        with col2:
            st.markdown('<span class="section-tag">Conditions</span>', unsafe_allow_html=True)
            weather = st.selectbox(
                "Weather",
                ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog", "Windy"],
                format_func=lambda w: f"{WEATHER_EMOJI.get(w, '')} {w}",
            )
            traffic = st.selectbox(
                "Traffic Density",
                ["Low", "Medium", "High", "Jam"],
                format_func=lambda t: f"{TRAFFIC_EMOJI.get(t, '')} {t}",
            )
            festival = st.selectbox("Festival?", ["No", "Yes"])
            order_type = st.selectbox("Order Type", ["Snack", "Meal", "Drinks", "Buffet"])
            vehicle_type = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_scooter", "bicycle"])

        with col3:
            st.markdown('<span class="section-tag">Delivery Person</span>', unsafe_allow_html=True)
            age = st.number_input("Age", 15, 65, 30)
            ratings = st.number_input("Rating (0-5)", 0.0, 5.0, 4.5, step=0.1)
            vehicle_condition = st.selectbox("Vehicle Condition (0-3)", [0, 1, 2, 3], index=2)
            multiple_deliveries = st.selectbox("Multiple Deliveries", [0, 1, 2, 3])

        submit = st.form_submit_button("🔮 Predict Delivery Time", use_container_width=True)

    if submit:
        is_valid, error_msg = validate_inputs(
            distance_km, age, ratings, order_hour, multiple_deliveries, vehicle_condition
        )
        if not is_valid:
            st.error(error_msg)
        else:
            with st.spinner("Crunching the numbers..."):
                time.sleep(0.4)
                input_df = preprocess_single_row(
                    distance_km=distance_km, weather=weather, traffic=traffic,
                    order_hour=order_hour, city=city, order_type=order_type,
                    vehicle_type=vehicle_type, multiple_deliveries=multiple_deliveries,
                    festival=festival, age=age, ratings=ratings,
                    vehicle_condition=vehicle_condition,
                )
                pred_time = predict_delivery_time(model, input_df)
                margin = error_margin(model)

            st.balloons()

            res_col, gauge_col = st.columns([1, 1.2])

            with res_col:
                st.markdown(
                    f"""
                    <div class="result-card">
                        <p>Estimated Delivery Time</p>
                        <h2>{pred_time:.0f} min</h2>
                        <p>± {margin:.1f} min expected error margin</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.write("")
                input_df_out = input_df.copy()
                input_df_out["Predicted_Time_min"] = pred_time
                csv = input_df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Prediction Data (CSV)",
                    data=csv,
                    file_name="delivery_prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with gauge_col:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pred_time,
                    number={"suffix": " min", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 60], "tickwidth": 1},
                        "bar": {"color": "#F09819"},
                        "steps": [
                            {"range": [0, 20], "color": "#d4f4dd"},
                            {"range": [20, 35], "color": "#ffe8b3"},
                            {"range": [35, 60], "color": "#ffd0c2"},
                        ],
                        "threshold": {
                            "line": {"color": "#FF512F", "width": 4},
                            "thickness": 0.8,
                            "value": pred_time,
                        },
                    },
                ))
                gauge.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(gauge, use_container_width=True)

            st.markdown("### 🗺️ Simulated Route")
            base_lat, base_lon = 12.9716, 77.5946
            dest_lat = base_lat + distance_km * 0.009
            dest_lon = base_lon + distance_km * 0.009

            m = folium.Map(location=[(base_lat + dest_lat) / 2, (base_lon + dest_lon) / 2], zoom_start=12, tiles="CartoDB positron")
            folium.Marker(
                [base_lat, base_lon], popup="Restaurant",
                icon=folium.Icon(color="orange", icon="cutlery", prefix="fa"),
            ).add_to(m)
            folium.Marker(
                [dest_lat, dest_lon], popup="Delivery Location",
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
            ).add_to(m)
            folium.PolyLine(
                [[base_lat, base_lon], [dest_lat, dest_lon]],
                color="#F09819", weight=4, opacity=0.8, dash_array="8",
            ).add_to(m)
            st_folium(m, width=None, height=400, returned_objects=[], use_container_width=True)
            st.caption("Route is illustrative — a straight line scaled to the entered distance, not an actual road path.")


# --------------------------------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------------------------------
elif page == "📊 Dashboard":
    st.markdown("""
    <div class="hero">
        <h1>📊 Model Performance Dashboard</h1>
        <p>How the deployed model was chosen, and what the data looks like underneath it.</p>
    </div>
    """, unsafe_allow_html=True)

    if not MODEL_LOADED:
        st.stop()

    try:
        df = pd.read_csv("data/processed/food_delivery_processed.csv")
    except Exception:
        df = None

    st.markdown('<span class="section-tag">Final Model Metrics</span>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{model.metrics['r2']:.3f}")
    col2.metric("RMSE", f"{model.metrics['rmse']:.2f} min")
    col3.metric("MAE", f"{model.metrics['mae']:.2f} min")
    col4.metric("MSE", f"{model.metrics['mse']:.2f}")

    st.markdown("---")
    st.markdown('<span class="section-tag">Baseline vs Deployed Models</span>', unsafe_allow_html=True)

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest", "HistGradientBoosting"],
        "R²":    [0.577, 0.820, 0.822],
        "RMSE":  [5.892, 3.845, 3.820],
        "MAE":   [4.709, 3.058, 3.044],
    })

    def highlight_winner(row):
        is_winner = row["Model"] == type(model.model).__name__ or row["R²"] == comparison_df["R²"].max()
        return ["background-color: rgba(56, 239, 125, 0.18); font-weight: 700;" if is_winner else "" for _ in row]

    st.dataframe(
        comparison_df.style.apply(highlight_winner, axis=1).format({"R²": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("🏆 Highlighted row is the model deployed in this app.")

    st.markdown("---")
    if df is not None:
        st.markdown('<span class="section-tag">Processed Dataset Preview</span>', unsafe_allow_html=True)
        st.write(f"Rows: **{df.shape[0]:,}** | Columns: **{df.shape[1]}**")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="section-tag">EDA & Results Plots</span>', unsafe_allow_html=True)

    plot_files = sorted(
        [f for f in os.listdir("assets/plots") if f.endswith(".png")]
    ) if os.path.isdir("assets/plots") else []

    if not plot_files:
        st.warning("No plots found. Run `python src/visualize.py` first.")
    else:
        tab_labels = [f.replace(".png", "").replace("_", " ").title() for f in plot_files]
        tabs = st.tabs(tab_labels)
        for tab, fname in zip(tabs, plot_files):
            with tab:
                st.image(os.path.join("assets/plots", fname), use_container_width=True)
