from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
DATA_PATH = Path(__file__).resolve().parent / "Agartala_AQIBulletins.csv"
WINDOW_SIZE = 7


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def summarize_missing_and_outliers(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    missing_summary = df.isna().sum()

    q1 = df["Index Value"].quantile(0.25)
    q3 = df["Index Value"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df["Index Value"] < lower_bound) | (df["Index Value"] > upper_bound)]

    return missing_summary, outliers[["date", "Index Value"]]


def plot_time_series(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(df["date"], df["Index Value"], marker="o", linewidth=1, markersize=3)
    plt.title("Index Value over Time")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_series_index_value.png", dpi=200)
    plt.close()


def plot_pollutant_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x="Prominent Pollutant", order=df["Prominent Pollutant"].value_counts().index)
    plt.title("Prominent Pollutant Distribution")
    plt.xlabel("Prominent Pollutant")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "prominent_pollutant_distribution.png", dpi=200)
    plt.close()


def create_windowed_features(series: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(window, len(series)):
        X.append(series.iloc[idx - window: idx].values)
        y.append(series.iloc[idx])
    return np.array(X), np.array(y)


@dataclass
class ForecastResults:
    mae: float
    rmse: float
    r2: float
    predictions: np.ndarray
    actuals: np.ndarray


def train_forecasting_model(df: pd.DataFrame, window: int) -> ForecastResults:
    series = df["Index Value"]
    X, y = create_windowed_features(series, window)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=6,
        random_state=42,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    # Compute RMSE manually for compatibility with older sklearn signatures
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, preds)

    return ForecastResults(mae, rmse, r2, preds, y_test)


def plot_forecast_results(results: ForecastResults) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(results.actuals, label="Actual", marker="o", linewidth=1)
    plt.plot(results.predictions, label="Predicted", marker="o", linewidth=1)
    plt.title("Next-Day Index Value Forecast")
    plt.xlabel("Test Sample Index")
    plt.ylabel("Index Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "forecast_actual_vs_pred.png", dpi=200)
    plt.close()


def cluster_pollution(df: pd.DataFrame) -> pd.DataFrame:
    model = KMeans(n_clusters=3, random_state=42, n_init="auto")
    df = df.copy()
    df["cluster"] = model.fit_predict(df[["Index Value"]])

    centers = model.cluster_centers_.flatten()
    ordered_clusters = np.argsort(centers)
    labels = {ordered_clusters[0]: "Low", ordered_clusters[1]: "Medium", ordered_clusters[2]: "High"}
    df["pollution_level"] = df["cluster"].map(labels)
    return df


def plot_clusters(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 4))
    sns.scatterplot(data=df, x="date", y="Index Value", hue="pollution_level", palette="viridis")
    plt.title("Pollution Level Clusters Over Time")
    plt.xlabel("Date")
    plt.ylabel("Index Value")
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cluster_over_time.png", dpi=200)
    plt.close()


def run_full_analysis() -> dict:
    """
    Run the full pipeline and return results for use in a web frontend.

    Returns a dictionary with:
      - missing_summary: pd.Series
      - outliers: pd.DataFrame
      - forecast_results: ForecastResults
      - cluster_counts: pd.Series
      - cluster_summary: pd.DataFrame
      - figure_files: list of figure filenames
    """
    df = load_data(DATA_PATH)

    missing_summary, outlier_rows = summarize_missing_and_outliers(df)

    # EDA plots
    plot_time_series(df)
    plot_pollutant_distribution(df)

    # Forecast
    forecast_results = train_forecasting_model(df, WINDOW_SIZE)
    plot_forecast_results(forecast_results)

    # Clustering
    clustered_df = cluster_pollution(df)
    plot_clusters(clustered_df)
    cluster_counts = clustered_df["pollution_level"].value_counts()
    cluster_summary = (
        clustered_df.groupby("pollution_level")["Index Value"]
        .agg(["min", "max", "mean"])
        .sort_values("mean")
    )

    figure_files = [
        "time_series_index_value.png",
        "prominent_pollutant_distribution.png",
        "forecast_actual_vs_pred.png",
        "cluster_over_time.png",
    ]

    return {
        "missing_summary": missing_summary,
        "outliers": outlier_rows,
        "forecast_results": forecast_results,
        "cluster_counts": cluster_counts,
        "cluster_summary": cluster_summary,
        "figure_files": figure_files,
    }


def main() -> None:
    df = load_data(DATA_PATH)

    missing_summary, outlier_rows = summarize_missing_and_outliers(df)
    print("Missing values per column:")
    print(missing_summary)
    if outlier_rows.empty:
        print("No abnormal Index Value readings detected via IQR rule.")
    else:
        print("Potential abnormal Index Value readings:")
        print(outlier_rows.to_string(index=False))

    plot_time_series(df)
    plot_pollutant_distribution(df)

    forecast_results = train_forecasting_model(df, WINDOW_SIZE)
    print(f"Forecast MAE: {forecast_results.mae:.2f}")
    print(f"Forecast RMSE: {forecast_results.rmse:.2f}")
    print(f"Forecast R^2: {forecast_results.r2:.2f}")
    plot_forecast_results(forecast_results)

    clustered_df = cluster_pollution(df)
    plot_clusters(clustered_df)
    print("\nCluster distribution:")
    print(clustered_df["pollution_level"].value_counts())
    print("\nCluster interpretation by Index Value ranges:")
    summary = clustered_df.groupby("pollution_level")["Index Value"].agg(["min", "max", "mean"]).sort_values("mean")
    print(summary)


if __name__ == "__main__":
    main()

