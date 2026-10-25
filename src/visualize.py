"""
src/visualize.py
------------------
Loads the processed dataset (and predictions) and generates
all EDA plots into assets/plots/.

Run:
    python src/visualize.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") # Headless mode (prevents GUI windows from popping up)
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_FILE = os.path.join("data", "processed", "food_delivery_processed.csv")
PREDICTIONS_FILE = os.path.join("assets", "predictions.csv")
PLOTS_DIR = os.path.join("assets", "plots")

# Set a nice visual style
sns.set_style("whitegrid")

def savefig(name):
    """Helper to save the current plot and close it."""
    path = os.path.join(PLOTS_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved -> {path}")

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(9,7))
    sns.heatmap(numeric_df.corr(),annot = True, fmt = ".2f", cmap = "coolwarm", square=True)
    plt.title("Correlation Heatmap")
    savefig("01_correlation_heatmap.png")

def plot_delivery_time_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Time_taken(min)"], kde=True, bins=25, color="teal")
    plt.title("Delivery Time Distribution")
    plt.xlabel("Time taken (min)")
    savefig("02_delivery_time_distribution.png")

def plot_distance_vs_time(df):
    plt.figure(figsize=(8, 6))
    hue_col = "Weatherconditions" if "Weatherconditions" in df.columns else None
    sns.scatterplot(data=df, x="distance_km", y="Time_taken(min)", hue=hue_col, alpha=0.7)
    plt.title("Distance vs Delivery Time")
    plt.xlabel("Distance (km)")
    plt.ylabel("Time taken (min)")
    savefig("03_distance_vs_time.png")

def plot_time_by_weather(df):
    if "Weatherconditions" not in df.columns:
        return
    plt.figure(figsize=(9, 5))
    order = df.groupby("Weatherconditions")["Time_taken(min)"].median().sort_values().index
    sns.boxplot(data=df, x="Weatherconditions", y="Time_taken(min)", order=order)
    plt.title("Delivery Time by Weather Condition")
    plt.xticks(rotation=30)
    savefig("04_time_by_weather_boxplot.png")

def plot_time_by_traffic(df):
    if "Road_traffic_density" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    sns.violinplot(data=df, x="Road_traffic_density", y="Time_taken(min)")
    plt.title("Delivery Time by Traffic Density")
    savefig("05_time_by_traffic_violin.png")

def plot_avg_time_by_city(df):
    if "City" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    avg = df.groupby("City")["Time_taken(min)"].mean().sort_values()
    sns.barplot(x=avg.index, y=avg.values, hue=avg.index, palette="viridis", legend=False)
    plt.title("Average Delivery Time by City")
    plt.ylabel("Avg time (min)")
    savefig("06_avg_time_by_city.png")

def plot_hourly_time_series(df):
    if "order_hour" not in df.columns:
        return
    plt.figure(figsize=(9, 5))
    hourly = df.groupby("order_hour")["Time_taken(min)"].mean()
    sns.lineplot(x=hourly.index, y=hourly.values, marker="o", color="darkorange")
    plt.title("Average Delivery Time by Order Hour")
    plt.xlabel("Order hour (24h)")
    plt.ylabel("Avg time (min)")
    savefig("09_hourly_time_series.png")

def plot_pairplot(df):
    cols = [c for c in ["distance_km", "Time_taken(min)", "Delivery_person_Age",
                         "Delivery_person_Ratings", "order_hour"] if c in df.columns]
    if len(cols) < 2:
        return
    sample = df[cols].sample(min(300, len(df)), random_state=42)
    g = sns.pairplot(sample, diag_kind="kde", plot_kws={"alpha": 0.6, "s": 20})
    g.fig.suptitle("Pairplot of Key Numeric Features", y=1.02)
    g.savefig(os.path.join(PLOTS_DIR, "07_pairplot.png"), dpi=120)
    plt.close("all")
    print(f"Saved -> {os.path.join(PLOTS_DIR, '07_pairplot.png')}")

def plot_outlier_boxplots(df):
    cols = [c for c in ["distance_km", "Delivery_person_Age", "Time_taken(min)"] if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        sns.boxplot(y=df[c], ax=ax, color="salmon")
        ax.set_title(c)
    plt.suptitle("Outlier Boxplots (post-cleaning)")
    savefig("08_outlier_boxplots.png")

def plot_actual_vs_predicted():
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"Skipping actual-vs-predicted plot: {PREDICTIONS_FILE} not found.")
        return
    preds = pd.read_csv(PREDICTIONS_FILE)
    plt.figure(figsize=(7, 7))
    plt.scatter(preds["Actual"], preds["Predicted"], alpha=0.6, color="steelblue")
    lims = [preds[["Actual", "Predicted"]].min().min(), preds[["Actual", "Predicted"]].max().max()]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual (min)")
    plt.ylabel("Predicted (min)")
    plt.title("Actual vs Predicted Delivery Time")
    plt.legend()
    savefig("10_actual_vs_predicted.png")

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(PROCESSED_FILE)
    print(f"Loaded processed data: {df.shape}")

    plot_correlation_heatmap(df)
    plot_delivery_time_distribution(df)
    plot_distance_vs_time(df)
    plot_time_by_weather(df)
    plot_time_by_traffic(df)
    plot_avg_time_by_city(df)
    plot_pairplot(df)
    plot_outlier_boxplots(df)
    plot_hourly_time_series(df)
    plot_actual_vs_predicted()

    print("\nAll plots generated in", PLOTS_DIR)

if __name__ == "__main__":
    main()

# =============================================================================
# VISUALIZATION RATIONALE (Why we chose these specific graphs)
# =============================================================================
# 01. Correlation Heatmap:
#     Used to see linear relationships. Tells us which numeric features 
#     (like distance_km) have the strongest positive/negative impact on time.
#
# 02. Delivery Time Distribution (Histogram):
#     Used to check the target variable. We want to confirm it resembles a 
#     normal "bell curve" and isn't heavily skewed (which would confuse models).
#
# 03. Distance vs Time (Scatterplot):
#     Used to visually prove the core hypothesis: longer distances take more time. 
#     Hue (color) is added for weather to see if storms cause vertical clusters.
#
# 04. Delivery Time by Weather (Boxplot):
#     Used to compare categories. Boxplots are perfect for showing the median,
#     the spread (IQR), and outliers across different weather conditions.
#
# 05. Delivery Time by Traffic (Violin Plot):
#     Similar to a boxplot, but shows the "density" of the data. We use it here
#     to see exactly where most deliveries cluster under Low/Medium/High/Jam traffic.
#
# 06. Average Time by City (Bar Chart):
#     Used for clear categorical comparison. Instantly shows which city's 
#     logistics network is the slowest on average.
#
# 07. Pairplot:
#     Used to visualize the pairwise relationships between 5 key numeric variables
#     simultaneously. Great for a report to show overall data structure.
#
# 08. Outlier Boxplots:
#     Used to prove that our preprocess.py step worked. The boxes are tight 
#     and contain almost no extreme dots, meaning the 1.5xIQR filter succeeded.
#
# 09. Hourly Time Series (Line Chart):
#     Used to show temporal trends. Line charts are best for time data, helping 
#     us spot rush hours (e.g., lunch and dinner spikes in delivery times).
#
# 10. Actual vs Predicted (Scatterplot):
#     THE MOST IMPORTANT PLOT. We compare the true test values to what the 
#     model guessed. The closer the dots are to the red dashed line, the better 
#     the model. (Our Random Forest shows a tight, strong diagonal line).
# =============================================================================