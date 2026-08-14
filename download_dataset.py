import os

try:
    import pandas as pd
except ImportError:
    pd = None

# Try kagglehub first; if not available, fall back to kaggle API
try:
    import kagglehub as kh
except Exception:
    kh = None

try:
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
except Exception:
    KaggleApi = None

def download_and_save():
    try:
        print("Downloading dataset from Kaggle...")
        # Use kagglehub if available, otherwise fall back to kaggle API
        if kh is not None:
            path = kh.dataset_download("changlechangsu/india-food-delivery-time-prediction")
        elif KaggleApi is not None:
            api = KaggleApi()
            api.authenticate()
            # download and unzip into a temporary folder
            tmp_path = "data_tmp"
            api.dataset_download_files("changlechangsu/india-food-delivery-time-prediction", path=tmp_path, unzip=True)
            path = tmp_path
        else:
            raise RuntimeError("Neither kagglehub nor KaggleApi is available.")
        print(f"Dataset downloaded to: {path}")

        # Find the first CSV file in the downloaded folder
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            # The dataset might be a text file, check for .txt as well
            csv_files = [f for f in os.listdir(path) if f.endswith('.txt')]
            if not csv_files:
                raise FileNotFoundError("No CSV or TXT file found in the downloaded dataset.")

        file_path = os.path.join(path, csv_files[0])
        # The file might be a .txt file, but pandas can still read it if it's structured like a CSV
        df = pd.read_csv(file_path)

        # Save to our data folder
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/food_delivery_dataset.csv", index=False)

        print(f"Dataset saved to data/food_delivery_dataset.csv")
        print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
        print(f"Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error: {e}")
        print("\nIf download fails, manually download the CSV from Kaggle and place it at:")
        print("   data/food_delivery_dataset.csv")

if __name__ == "__main__":
    download_and_save()