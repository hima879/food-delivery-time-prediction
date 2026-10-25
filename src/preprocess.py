"""
src/preprocess.py
------------------
Clean the raw Kaggle food-delivery dataset and engineer features.

Run from the project root:
    python src/preprocess.py

Input : the first .csv / .json file in data/raw/
Output: data/processed/food_delivery_processed.csv
"""
# Imports and Setup
import os
import glob
import json
import re
import pandas as pd
import numpy as np

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "food_delivery_processed.csv")

# Finding and Loading the Dataset
def find_raw_file() -> str:
    """Locate the first CSV/JSON/TXT file in data/raw/ and return its path."""
    candidates = (
        glob.glob(os.path.join(RAW_DIR, "*.csv"))
        + glob.glob(os.path.join(RAW_DIR, "*.json"))
        + glob.glob(os.path.join(RAW_DIR, "*.txt"))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No CSV/JSON/TXT file found in '{RAW_DIR}/'. "
            "Run `python download_data.py` first, or place the dataset there manually."
        )
    for c in candidates:
        name = os.path.basename(c).lower()
        if "train" in name or "delivery" in name:
            return c
    return candidates[0]

def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV or JSON file."""
    # Peek at the first 2048 bytes to see if it's JSON
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.read(2048).lstrip()

    # JSON files start with '{' (object) or '[' (array), CSV files usually start with a header line
    looks_like_json = path.endswith(".json") or head.startswith("{") or head.startswith("[")

    if looks_like_json:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(path)
    return df


# String Cleaning & Target Parsing
def clean_columns_and_strings(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        # Strip spaces from the left and right
        df[c] = df[c].astype(str).str.strip()
        # Turn literal "nan" strings into real Python None/NaN
        df[c] = df[c].replace({"nan": np.nan, "NaN": np.nan, "Nan": np.nan, "": np.nan})
    return df


def clean_target_column(df: pd.DataFrame) -> pd.DataFrame:
    target_col = None
    for c in df.columns:
        if "Time_taken" in c:
            target_col = c
            break

    def extract_minutes(val):
        if pd.isna(val):
            return np.nan
        # Regex looks for digits and extracts them
        match = re.search(r"(\d+)", str(val))
        return int(match.group(1)) if match else np.nan

    df["Time_taken(min)"] = df[target_col].apply(extract_minutes)
    if target_col != "Time_taken(min)":
        df = df.drop(columns=[target_col])
    return df

# Numeric Conversion & Feature Engineering
NUMERIC_LIKE_COLS = [
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Vehicle_condition", "multiple_deliveries",
]

def coerce_numeric_columns(df:pd.DataFrame)->pd.DataFrame:
    for c in NUMERIC_LIKE_COLS:
        if c in df.columns:
            # Turn "37" (string) into 37.0(float)
            df[c] = pd.to_numeric(df[c],errors="coerce")
    return df

def haversine_km(lat1,lon1,lat2,lon2):
    # Mathematical formula to get distance between two GPS points in KM
    R = 6371.0
    lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])
    dlat = lat2-lat1
    dlon = lon2-lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c

def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    lat1 = df["Restaurant_latitude"].astype(float)
    lon1 = df["Restaurant_longitude"].astype(float)
    lat2 = df["Delivery_location_latitude"].astype(float)
    lon2 = df["Delivery_location_longitude"].astype(float)
    # .abs() ensures we don't get negative distances if coordinates are flipped
    df["distance_km"] = haversine_km(lat1, lon1, lat2, lon2).abs()
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    def parse_hour(val):
        if pd.isna(val):
            return np.nan
        m = re.match(r"^(\d{1,2}):", str(val).strip()) # Extracts "17" from "17:30:00"
        return int(m.group(1)) if m else np.nan

    if "Time_Orderd" in df.columns:
        df["order_hour"] = df["Time_Orderd"].apply(parse_hour)
        df["order_hour"] = df["order_hour"].fillna(df["order_hour"].median())

    if "Order_Date" in df.columns:
        parsed = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y", errors="coerce")
        df["weekday"] = parsed.dt.day_name()
        df["weekday"] = df["weekday"].fillna("Unknown")
    return df

# Missing Values and Outliers
def handle_missing_values(df:pd.DataFrame)->pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include="object").columns

    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median()) # Fill numbers with median

    for c in categorical_cols:
        if df[c].isna().any():
            mode_val = df[c].mode(dropna=True)  # Fill text with most frequent 
            df[c] = df[c].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    return df

def remove_outliers_iqr(df:pd.DataFrame,cols)->pd.DataFrame:
    before = len(df)
    for c in cols:
        if c not in df.columns: continue
        # IQR = Interquartile Range (Middle 50% of data)
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        # Drop anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[c] >= lower) & (df[c] <= upper)]
    after = len(df)
    print(f"Outlier removal: {before} -> {after} rows ({before - after} removed)")
    return df

# The Main Pipeline
def clean_weather_column(df: pd.DataFrame) -> pd.DataFrame:
    weather_col = None
    for c in df.columns:
        if "Weather" in c:
            weather_col = c
            break
    if weather_col:
        df[weather_col] = df[weather_col].astype(str).str.replace("conditions ", "", regex=False).str.strip()
        df = df.rename(columns={weather_col: "Weatherconditions"})
    return df

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load
    raw_path = find_raw_file()
    df = load_raw_data(raw_path)

    # 2. Clean
    df = clean_columns_and_strings(df)
    df = clean_target_column(df)
    df = coerce_numeric_columns(df)

    # 3. Engineer Features
    df = add_distance_feature(df)
    df = add_time_features(df)
    df = clean_weather_column(df)

    # 4. Handle Missing & Outliers
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, ["distance_km", "Delivery_person_Age", "Time_taken(min)"])

    # 5. Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nProcessed data saved to: {OUTPUT_FILE}")
    print(f"Final shape: {df.shape}")

    n_train = int(len(df) * 0.8)
    n_test = len(df) - n_train
    print(f"Train/test split (80/20): train={n_train} rows, test={n_test} rows")

if __name__ == "__main__":
    main()

"""
src/preprocess.py
------------------
Loads the raw food delivery dataset (CSV or JSON), cleans it, engineers
features, removes outliers, and saves the processed dataset to
data/processed/food_delivery_processed.csv.

Run:
    python src/preprocess.py

Preprocessing Steps:
1. Load raw file (auto-detects JSON arrays disguised as .csv files).
2. Clean strings (strip trailing spaces) and convert literal "NaN" strings to real nulls.
3. Parse target "Time_taken(min)" (e.g. extracts integer 24 from "(min) 24").
4. Coerce numeric columns from strings to floats (Age, Ratings, Lat/Lon, etc).
5. Engineer Features:
   - distance_km (Haversine distance from the 4 GPS coordinate columns).
   - order_hour (Extract hour from "Time_Orderd").
   - weekday (Parsed from "Order_Date").
   - Clean weather strings ("conditions Sunny" -> "Sunny").
6. Impute missing values (median for numeric, mode for categorical).
7. Remove statistical outliers using 1.5 * IQR on Distance, Age, and Time.
8. Save the cleaned dataset to data/processed/.
"""