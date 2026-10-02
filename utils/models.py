"""
models.py — Model training, evaluation, and forecasting logic.

Implements four forecasting approaches and three clustering algorithms.
All model objects are saved/loaded via joblib for caching.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# SECTION 1: FORECASTING MODELS
# ═══════════════════════════════════════════════════════════════


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute forecasting metrics: MAE, RMSE, MAPE, sMAPE.

    MAPE can blow up when y_true has zeros, so we guard against
    divide-by-zero and also compute sMAPE as a more stable alternative.

    Parameters
    ----------
    y_true : array-like
        Actual values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Keys: MAE, RMSE, MAPE, sMAPE
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE — guard against zero denominator
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    # sMAPE — symmetric, more stable when values are near zero
    denom = np.abs(y_true) + np.abs(y_pred)
    smape_vals = np.where(denom == 0, 0, 2 * np.abs(y_true - y_pred) / denom)
    smape = np.mean(smape_vals) * 100

    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2), "sMAPE": round(smape, 2)}


def seasonal_naive_forecast(train: pd.Series, horizon: int, period: int = 12) -> np.ndarray:
    """
    Seasonal naive baseline: predict by repeating values from
    the same month one year ago (period=12 for monthly data).

    This is the simplest reasonable baseline for monthly data.
    If the naive baseline beats ML models, it means the series
    is too noisy or short for more complex models to learn from.
    """
    forecast = []
    values = train.values
    for h in range(horizon):
        # Use the value from 'period' steps back, cycling if needed
        idx = len(values) - period + (h % period)
        if idx >= 0:
            forecast.append(values[idx])
        else:
            forecast.append(values[-1])  # fallback to last known value
    return np.array(forecast)


def train_sarima(train_series: pd.Series, horizon: int) -> tuple:
    """
    Fit a SARIMA(1,1,1)(1,1,1,12) model and forecast `horizon` steps.

    SARIMA is chosen because:
    - The series may have monthly seasonality (period=12)
    - Differencing (d=1) handles any trend
    - Low orders (p=1, q=1) to avoid overfitting on this short series

    Returns (forecast_array, fitted_model)
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    # Use modest order to avoid overfitting on ~120 training points
    try:
        model = SARIMAX(
            train_series,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=200)
        forecast = fitted.forecast(steps=horizon)
        # Clamp negatives to zero (incident counts can't be negative)
        forecast = np.maximum(forecast.values, 0)
        return forecast, fitted
    except Exception as e:
        print(f"SARIMA failed: {e}")
        return np.full(horizon, train_series.mean()), None


def train_prophet(train_df: pd.DataFrame, horizon: int) -> tuple:
    """
    Fit Facebook Prophet and forecast `horizon` steps.

    Prophet is robust to outliers and missing data, and
    automatically handles yearly seasonality.
    We pass the 'month' column as 'ds' and 'incident_count' as 'y'.

    Returns (forecast_df, model)
    """
    from prophet import Prophet

    # Prepare data in Prophet's expected format
    prophet_df = train_df[["month", "incident_count"]].rename(
        columns={"month": "ds", "incident_count": "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,  # conservative to avoid overfitting
    )
    model.fit(prophet_df)

    # Create future dates for forecasting
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)

    # Extract only the forecast portion (last `horizon` rows)
    forecast_values = forecast.tail(horizon)["yhat"].values
    forecast_values = np.maximum(forecast_values, 0)

    return forecast_values, forecast


def train_xgboost(train_features: pd.DataFrame, train_target: pd.Series,
                  feature_cols: list) -> object:
    """
    Train an XGBoost regressor on lag/rolling features.

    XGBoost is included because it can capture nonlinear
    relationships between lag features and the target. However,
    for noisy short series it may not beat simpler methods.

    Returns the fitted XGBRegressor model.
    """
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        verbosity=0,
    )
    model.fit(train_features[feature_cols], train_target)
    return model


def xgboost_recursive_forecast(model, last_known: pd.DataFrame,
                                feature_cols: list, horizon: int) -> np.ndarray:
    """
    Generate multi-step XGBoost forecasts using recursive (iterated) strategy.

    At each step, we predict the next value, then update the lag
    features to include that prediction for the following step.
    This propagates uncertainty, which is a known limitation.
    """
    predictions = []
    current = last_known.copy()

    for step in range(horizon):
        pred = model.predict(current[feature_cols].values.reshape(1, -1))[0]
        pred = max(pred, 0)  # Clamp to zero
        predictions.append(pred)

        # Shift lags forward for the next prediction
        new_row = current.copy()
        new_row["lag_3"] = current["lag_2"].values[0]
        new_row["lag_2"] = current["lag_1"].values[0]
        new_row["lag_1"] = pred

        # Update rolling means approximately
        new_row["rolling_mean_3"] = np.mean([pred, current["lag_1"].values[0],
                                              current["lag_2"].values[0]])
        new_row["rolling_mean_6"] = (current["rolling_mean_6"].values[0] * 5 + pred) / 6

        # Advance month number
        current_month = int(current["month_number"].values[0])
        new_row["month_number"] = (current_month % 12) + 1

        current = new_row

    return np.array(predictions)


# ═══════════════════════════════════════════════════════════════
# SECTION 2: CLUSTERING MODELS
# ═══════════════════════════════════════════════════════════════


def run_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple:
    """
    Fit K-Means clustering.

    K-Means is our primary clustering algorithm because:
    - It scales well to ~10k records
    - Produces interpretable, balanced clusters
    - k is tunable via elbow method and silhouette score

    Returns (labels, model, silhouette, davies_bouldin)
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    db = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float("inf")

    return labels, model, round(sil, 4), round(db, 4)


def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> tuple:
    """
    Fit DBSCAN (Density-Based Spatial Clustering).

    DBSCAN doesn't require specifying k in advance, and can find
    arbitrarily shaped clusters. However, it may label many points
    as noise (-1) on this sparse binary feature space.

    Returns (labels, model, silhouette, davies_bouldin)
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        # Exclude noise points for silhouette calculation
        mask = labels != -1
        if mask.sum() > 1 and len(set(labels[mask])) > 1:
            sil = silhouette_score(X[mask], labels[mask])
            db = davies_bouldin_score(X[mask], labels[mask])
        else:
            sil = -1
            db = float("inf")
    else:
        sil = -1
        db = float("inf")

    return labels, model, round(sil, 4), round(db, 4)


def run_agglomerative(X: np.ndarray, n_clusters: int) -> tuple:
    """
    Fit Agglomerative (hierarchical) clustering.

    This bottom-up approach merges the closest pairs of clusters
    iteratively. Useful as a comparison but slower than K-Means.

    Returns (labels, model, silhouette, davies_bouldin)
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    db = davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float("inf")

    return labels, model, round(sil, 4), round(db, 4)


def compute_elbow(X: np.ndarray, k_range: range = range(2, 11)) -> tuple:
    """
    Compute the elbow method data: inertia and silhouette score
    for each candidate k value.

    Returns (k_values, inertias, silhouettes)
    """
    inertias = []
    silhouettes = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, labels))

    return k_values, inertias, silhouettes


def run_pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Reduce feature matrix to 2D using PCA for visualization.

    PCA is used ONLY for visualization, not for clustering.
    The clustering is done on the full feature space.

    Returns the 2-component transformed array.
    """
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


def generate_cluster_profiles(df: pd.DataFrame, labels: np.ndarray,
                               feature_names: list) -> list:
    """
    Auto-generate a human-readable profile for each cluster.

    For each cluster, identify:
    - Size (number of incidents)
    - Top action type (most common action_*)
    - Top actor type (most common actor_*)
    - Top attack pattern (most common pattern_*)
    - Top industries (most common sector_*)
    - A human-friendly cluster name

    Parameters
    ----------
    df : pd.DataFrame
        Original clustering data (for interpreting results).
    labels : np.ndarray
        Cluster labels assigned to each row.
    feature_names : list
        Names of the features used in clustering.

    Returns
    -------
    list of dict
        Each dict contains the cluster profile information.
    """
    from utils.data_loading import get_action_columns, get_actor_columns, get_pattern_columns, NAICS_MAP

    action_cols = get_action_columns()
    actor_cols = get_actor_columns()
    pattern_cols = get_pattern_columns()

    profiles = []
    unique_labels = sorted(set(labels))
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Skip noise cluster from DBSCAN

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_df = df[mask]
        size = mask.sum()

        # Top action — which action type is most prevalent in this cluster
        action_sums = cluster_df[action_cols].sum()
        top_action = action_sums.idxmax().replace("action_", "") if action_sums.max() > 0 else "Unknown"

        # Top actor — who is behind these incidents
        actor_sums = cluster_df[actor_cols].sum()
        top_actor = actor_sums.idxmax().replace("actor_", "") if actor_sums.max() > 0 else "Unknown"

        # Top pattern — VERIS pattern classification
        pattern_sums = cluster_df[pattern_cols].sum()
        top_pattern = pattern_sums.idxmax().replace("pattern_", "").replace("_", " ") if pattern_sums.max() > 0 else "Unknown"

        # Top industries — map sector codes to names
        if "industry_sector" in cluster_df.columns:
            sector_counts = cluster_df["industry_sector"].value_counts().head(3)
            top_industries = [
                NAICS_MAP.get(str(s).zfill(2), f"Sector {s}")
                for s in sector_counts.index
            ]
        else:
            top_industries = ["Unknown"]

        # Generate a human-friendly cluster name
        name = f"{top_industries[0]} {top_actor.lower()} {top_action.lower()}"
        name = name.title()

        profiles.append({
            "cluster_id": cluster_id,
            "size": size,
            "top_action": top_action,
            "top_actor": top_actor,
            "top_pattern": top_pattern,
            "top_industries": top_industries,
            "name": name,
            "action_distribution": action_sums.to_dict(),
            "actor_distribution": actor_sums.to_dict(),
            "pattern_distribution": pattern_sums.to_dict(),
        })

    return profiles


def predict_nearest_cluster(model, X_full, labels, query_features, feature_names):
    """
    Given user-selected features (industry, actor, action types),
    predict the nearest cluster and return the profile.

    Uses the K-Means model's cluster centers to find the nearest
    cluster to the query point.
    """
    # Build a query vector matching the feature space
    query_vector = np.zeros(len(feature_names))
    for i, fname in enumerate(feature_names):
        if fname in query_features:
            query_vector[i] = query_features[fname]

    # Predict cluster using the model
    if hasattr(model, "predict"):
        cluster = model.predict(query_vector.reshape(1, -1))[0]
    else:
        # Fallback: find nearest centroid manually
        from sklearn.metrics import pairwise_distances
        centroids = np.array([
            X_full[labels == c].mean(axis=0)
            for c in sorted(set(labels)) if c != -1
        ])
        dists = pairwise_distances(query_vector.reshape(1, -1), centroids)
        cluster = sorted(set(labels))[np.argmin(dists)]

    return cluster
