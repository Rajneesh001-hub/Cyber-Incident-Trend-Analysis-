"""
train.py — Offline model training script.

Trains all forecasting and clustering models, evaluates them,
and saves results to the models/ directory using joblib.

Usage:
    python train.py

This script can be run independently of Streamlit to pre-compute
and cache model artifacts. The Streamlit app can also train models
on-the-fly using st.cache_data.
"""

import warnings
import sys
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root to path so we can import utils
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.data_loading import load_forecasting_data, load_clustering_data
from utils.features import create_lag_features, get_xgb_feature_cols, prepare_clustering_features
from utils.models import (
    compute_metrics, seasonal_naive_forecast,
    train_sarima, train_prophet, train_xgboost,
    xgboost_recursive_forecast,
    run_kmeans, run_dbscan, run_agglomerative,
    compute_elbow, run_pca_2d, generate_cluster_profiles,
)

warnings.filterwarnings("ignore")

# Ensure models directory exists
MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_forecasting_models():
    """
    Train and evaluate all four forecasting models in both experiments:
    - Experiment A: raw data with the June 2023 spike
    - Experiment B: spike capped to median of neighbors

    Uses a strict time-based split: train through Dec 2021, test from Jan 2022.
    """
    print("=" * 60)
    print("TASK 1: FORECASTING MODEL TRAINING")
    print("=" * 60)

    for experiment, cap_spike in [("A_spike_included", False), ("B_spike_capped", True)]:
        print(f"\n{'─' * 50}")
        print(f"Experiment {experiment}")
        print(f"{'─' * 50}")

        # Load data (bypass Streamlit cache since we're running standalone)
        data_dir = Path(__file__).resolve().parent
        df = pd.read_csv(data_dir / "cyber_incidents_monthly_forecasting.csv")
        df["month"] = pd.to_datetime(df["month"])
        df = df.sort_values("month").reset_index(drop=True)

        if cap_spike:
            spike_idx = df[df["month"] == "2023-06-01"].index
            if len(spike_idx) > 0:
                idx = spike_idx[0]
                neighbors = []
                if idx > 0:
                    neighbors.append(df.loc[idx - 1, "incident_count"])
                if idx < len(df) - 1:
                    neighbors.append(df.loc[idx + 1, "incident_count"])
                df.loc[idx, "incident_count"] = int(np.median(neighbors))

        # Time-based split
        split_date = pd.Timestamp("2022-01-01")
        train_df = df[df["month"] < split_date].copy()
        test_df = df[df["month"] >= split_date].copy()

        train_series = train_df.set_index("month")["incident_count"]
        test_series = test_df.set_index("month")["incident_count"]
        test_horizon = len(test_series)

        print(f"  Train: {len(train_df)} months ({train_df['month'].min().strftime('%Y-%m')} to {train_df['month'].max().strftime('%Y-%m')})")
        print(f"  Test:  {len(test_df)} months ({test_df['month'].min().strftime('%Y-%m')} to {test_df['month'].max().strftime('%Y-%m')})")

        results = []

        # 1. Seasonal Naive
        print("\n  [1/4] Seasonal Naive Baseline...")
        naive_preds = seasonal_naive_forecast(train_series, test_horizon, period=12)
        naive_metrics = compute_metrics(test_series.values, naive_preds)
        naive_metrics["Model"] = "Seasonal Naive"
        results.append(naive_metrics)
        print(f"    MAE: {naive_metrics['MAE']}, RMSE: {naive_metrics['RMSE']}, sMAPE: {naive_metrics['sMAPE']}")

        # 2. SARIMA
        print("\n  [2/4] SARIMA(1,1,1)(1,1,1,12)...")
        sarima_preds, sarima_model = train_sarima(train_series, test_horizon)
        sarima_metrics = compute_metrics(test_series.values, sarima_preds)
        sarima_metrics["Model"] = "SARIMA"
        results.append(sarima_metrics)
        print(f"    MAE: {sarima_metrics['MAE']}, RMSE: {sarima_metrics['RMSE']}, sMAPE: {sarima_metrics['sMAPE']}")

        # 3. Prophet
        print("\n  [3/4] Prophet...")
        prophet_preds, prophet_forecast = train_prophet(train_df, test_horizon)
        prophet_metrics = compute_metrics(test_series.values, prophet_preds)
        prophet_metrics["Model"] = "Prophet"
        results.append(prophet_metrics)
        print(f"    MAE: {prophet_metrics['MAE']}, RMSE: {prophet_metrics['RMSE']}, sMAPE: {prophet_metrics['sMAPE']}")

        # 4. XGBoost with lag features
        print("\n  [4/4] XGBoost with lag features...")
        df_feat = create_lag_features(df)
        feat_cols = get_xgb_feature_cols()
        train_feat = df_feat[df_feat["month"] < split_date]
        test_feat = df_feat[df_feat["month"] >= split_date]

        if len(train_feat) > 0 and len(test_feat) > 0:
            xgb_model = train_xgboost(train_feat, train_feat["incident_count"], feat_cols)
            xgb_preds = xgb_model.predict(test_feat[feat_cols])
            xgb_preds = np.maximum(xgb_preds, 0)
            xgb_metrics = compute_metrics(test_feat["incident_count"].values, xgb_preds)
        else:
            xgb_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan}
            xgb_model = None

        xgb_metrics["Model"] = "XGBoost"
        results.append(xgb_metrics)
        print(f"    MAE: {xgb_metrics['MAE']}, RMSE: {xgb_metrics['RMSE']}, sMAPE: {xgb_metrics['sMAPE']}")

        # Save results
        results_df = pd.DataFrame(results)[["Model", "MAE", "RMSE", "MAPE", "sMAPE"]]
        results_df.to_csv(MODELS_DIR / f"forecast_results_{experiment}.csv", index=False)
        print(f"\n  Results saved to models/forecast_results_{experiment}.csv")

        # Save XGBoost model
        if xgb_model is not None:
            joblib.dump(xgb_model, MODELS_DIR / f"xgb_model_{experiment}.joblib")
            print(f"  XGBoost model saved to models/xgb_model_{experiment}.joblib")

        # Print comparison table
        print(f"\n  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'MAPE':>8} {'sMAPE':>8}")
        print(f"  {'─' * 57}")
        for _, row in results_df.iterrows():
            print(f"  {row['Model']:<25} {row['MAE']:>8.2f} {row['RMSE']:>8.2f} {row['MAPE']:>8.2f} {row['sMAPE']:>8.2f}")

        best = results_df.loc[results_df["MAE"].idxmin()]
        print(f"\n  🏆 Best model (lowest MAE): {best['Model']} (MAE = {best['MAE']:.2f})")


def train_clustering_models():
    """
    Train clustering models: K-Means (primary), DBSCAN, Agglomerative.
    Run with and without MOVEit events for comparison.
    """
    print("\n\n" + "=" * 60)
    print("TASK 2: CLUSTERING MODEL TRAINING")
    print("=" * 60)

    for setting, exclude_moveit in [("without_moveit", True), ("with_moveit", False)]:
        print(f"\n{'─' * 50}")
        print(f"Setting: {setting}")
        print(f"{'─' * 50}")

        # Load data
        data_dir = Path(__file__).resolve().parent
        df = pd.read_csv(
            data_dir / "cyber_incidents_clustering.csv",
            dtype={"industry_sector": str},
        )
        if exclude_moveit:
            df = df[df["is_2023_moveit_event"] == 0].copy()
        df = df.reset_index(drop=True)

        print(f"  Records: {len(df)}")

        X, feature_names = prepare_clustering_features(df)
        print(f"  Features: {len(feature_names)}")

        # Elbow method
        print("\n  Running elbow method (k=2..10)...")
        k_values, inertias, silhouettes = compute_elbow(X, range(2, 11))
        best_k = k_values[np.argmax(silhouettes)]
        print(f"  Best k by silhouette: {best_k} (score: {max(silhouettes):.4f})")

        # K-Means
        n_clusters = 5  # default, but use best_k for comparison
        print(f"\n  K-Means (k={n_clusters})...")
        km_labels, km_model, km_sil, km_db = run_kmeans(X, n_clusters)
        print(f"    Silhouette: {km_sil}, Davies-Bouldin: {km_db}")

        # Save K-Means model
        joblib.dump(km_model, MODELS_DIR / f"kmeans_{setting}.joblib")
        joblib.dump(feature_names, MODELS_DIR / f"feature_names_{setting}.joblib")

        # DBSCAN
        print(f"\n  DBSCAN (auto-tuned eps)...")
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        eps_val = float(np.percentile(distances[:, -1], 90))
        db_labels, _, db_sil, db_db = run_dbscan(X, eps=eps_val, min_samples=5)
        n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = (db_labels == -1).sum()
        print(f"    Clusters: {n_db_clusters}, Noise: {n_noise}, Silhouette: {db_sil}, D-B: {db_db}")

        # Agglomerative
        print(f"\n  Agglomerative (k={n_clusters})...")
        ag_labels, _, ag_sil, ag_db = run_agglomerative(X, n_clusters)
        print(f"    Silhouette: {ag_sil}, Davies-Bouldin: {ag_db}")

        # Generate cluster profiles
        print(f"\n  Cluster Profiles (K-Means):")
        profiles = generate_cluster_profiles(df, km_labels, feature_names)
        for p in profiles:
            print(f"    Cluster {p['cluster_id']}: {p['name']} ({p['size']} incidents)")
            print(f"      Action: {p['top_action']}, Actor: {p['top_actor']}, Pattern: {p['top_pattern']}")
            print(f"      Industries: {', '.join(p['top_industries'])}")

        # Comparison table
        print(f"\n  {'Algorithm':<20} {'Silhouette':>12} {'Davies-Bouldin':>15}")
        print(f"  {'─' * 47}")
        print(f"  {'K-Means':<20} {km_sil:>12.4f} {km_db:>15.4f}")
        print(f"  {'DBSCAN':<20} {db_sil:>12.4f} {db_db:>15.4f}")
        print(f"  {'Agglomerative':<20} {ag_sil:>12.4f} {ag_db:>15.4f}")

    print("\n\n✅ All models trained and saved successfully!")
    print(f"   Models directory: {MODELS_DIR}")


if __name__ == "__main__":
    print("🛡️ Cyber Incident Trend Analysis — Model Training")
    print("=" * 60)
    train_forecasting_models()
    train_clustering_models()
    print("\n🎉 Training complete! Run 'streamlit run app.py' to launch the dashboard.")
