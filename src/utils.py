"""
src/utils.py
-------------
Helper functions for the Streamlit app: loading the trained model, 
validating inputs, and formatting a single form submission into a 
model-ready DataFrame.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure we can import from the src folder when Streamlit runs
sys.path.append(os.path.dirname(__file__))
from model_train import DeliveryTimeModel, MODEL_DIR

def load_model(model_dir: str = MODEL_DIR) -> DeliveryTimeModel:
    """Load the trained model artifacts from the models/ folder."""
    return DeliveryTimeModel.load(model_dir)

def validate_inputs(distance_km, age, ratings, order_hour, multiple_deliveries, vehicle_condition):
    """Basic sanity checks on numeric form inputs. Returns (is_valid, message)."""
    if distance_km is None or distance_km <= 0:
        return False, "Distance must be a positive number."
    if not (15 <= age <= 65):
        return False, "Delivery person age should be between 15 and 65."
    if not (0 <= ratings <= 5):
        return False, "Ratings must be between 0 and 5."
    if not (0 <= order_hour <= 23):
        return False, "Order hour must be between 0 and 23."
    if not (0 <= multiple_deliveries <= 10):
        return False, "Multiple deliveries must be between 0 and 10."
    if not (0 <= vehicle_condition <= 3):
        return False, "Vehicle condition must be between 0 and 3."
    return True, ""

def preprocess_single_row(
    distance_km: float, weather: str, traffic: str, order_hour: int,
    city: str, order_type: str, vehicle_type: str, multiple_deliveries: int,
    festival: str, age: float = 30.0, ratings: float = 4.5, vehicle_condition: int = 1,
) -> pd.DataFrame:
    """Build a single-row DataFrame matching the exact columns the model expects."""
    row = {
        "Weatherconditions": weather,
        "Road_traffic_density": traffic,
        "Type_of_order": order_type,
        "Type_of_vehicle": vehicle_type,
        "City": city,
        "Festival": festival,
        "weekday": "Monday",  # Hardcoded for a single prediction (not in the form)
        "distance_km": distance_km,
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": ratings,
        "order_hour": order_hour,
        "multiple_deliveries": multiple_deliveries,
        "Vehicle_condition": vehicle_condition,
    }
    return pd.DataFrame([row])

def predict_delivery_time(model: DeliveryTimeModel, input_df: pd.DataFrame) -> float:
    """Run a single-row prediction and return a plain float."""
    pred = model.predict(input_df)
    return float(np.round(pred[0], 1))

def error_margin(model: DeliveryTimeModel) -> float:
    """Return the model's RMSE (used as a +/- error margin in the UI)."""
    return float(model.metrics.get("rmse", 0.0))

# ---------------------------------------------------------------------------
# FILE SUMMARY
# ---------------------------------------------------------------------------
# This module is the "glue layer" between the Streamlit UI and the trained
# delivery-time model. It keeps UI code clean by centralizing five concerns:
#
#   1. load_model()            - Loads serialized model artifacts from disk,
#                                so the app has one stable entry point.
#   2. validate_inputs()       - Sanity-checks raw form values before they
#                                reach the model, returning (ok, message) so
#                                the UI can show friendly errors instead of
#                                crashing on bad input.
#   3. preprocess_single_row() - Builds a one-row DataFrame whose columns
#                                exactly match the training schema, ensuring
#                                the model's encoders/scalers apply correctly.
#                                Optional fields fall back to sensible defaults
#                                and 'weekday' is hardcoded because the form
#                                does not collect it.
#   4. predict_delivery_time() - Runs a single prediction and returns a
#                                rounded plain float, safe to render in the UI.
#   5. error_margin()          - Exposes the model's RMSE so the UI can show
#                                predictions as "X minutes +/- RMSE", giving
#                                users an honest sense of accuracy.
#
# Design rationale: model training/loading lives in model_train.py; this file
# only handles input validation, schema alignment, and output formatting.
# Keeping these concerns separate makes both the model code and the UI code
# easier to test, maintain, and change independently.
# ---------------------------------------------------------------------------