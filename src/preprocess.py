import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import re

# ---- Configuration ----
NUMERIC_FEATURES = [
    'Delivery_person_Age', 'Delivery_person_Ratings', 'multiple_deliveries',
    'Vehicle_condition', 'distance_km', 'order_hour'
]
CATEGORICAL_FEATURES = [
    'Weatherconditions', 'Road_traffic_density', 'Type_of_order',
    'Type_of_vehicle', 'Festival', 'City'
]
TARGET = 'Time_taken(min)'
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---- Data Loading (handles JSON array) ----
def load_data(filepath='data/raw/India-Food-Delivery-Time-Prediction.csv'):
    """Load dataset from JSON or CSV; if JSON array, parse accordingly."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}.")
    
    # Try reading as JSON first (since content is a JSON array)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Loaded JSON data: {df.shape[0]} rows, {df.shape[1]} columns")
    except json.JSONDecodeError:
        # Fallback to CSV if JSON fails
        df = pd.read_csv(filepath)
        print(f"Loaded CSV data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Clean column names: strip whitespace
    df.columns = df.columns.str.strip()

    # Clean categorical string values (strip trailing spaces)
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Convert Time_taken(min) from "(min) 24" to integer 24
    if 'Time_taken(min)' in df.columns:
        df[TARGET] = df['Time_taken(min)'].str.extract(r'(\d+)').astype(float)
        df.drop(columns=['Time_taken(min)'], inplace=True)
    else:
        raise KeyError("Column 'Time_taken(min)' not found in dataset.")
    
    return df

# ---- Missing Values ----
def handle_missing_values(df):
    """Fill numeric with median, categorical with mode."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=np.number).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
    return df_clean