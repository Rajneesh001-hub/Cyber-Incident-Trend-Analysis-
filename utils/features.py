"""
features.py — Feature engineering for forecasting and clustering.

Creates lag features, rolling statistics, and one-hot encoded
industry sectors for the clustering pipeline.
"""

import pandas as pd
import numpy as np
from utils.data_loading import get_action_columns, get_actor_columns, get_pattern_columns, NAICS_MAP


def create_lag_features(df: pd.DataFrame, target_col: str = "incident_count") -> pd.DataFrame:
    """
    Create lag features for time-series forecasting with XGBoost.

    Lag features capture the autocorrelation in the series:
    - Lag 1, 2, 3: recent short-term momentum
    - Lag 12: same month last year (seasonality proxy)
    - Rolling mean 3 & 6: smoothed recent trends
    - Month number: captures seasonal patterns (1-12)

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'month' (datetime) and target_col columns.
    target_col : str
        The column to create lags from (default: 'incident_count').

    Returns
    -------
    pd.DataFrame
        Original columns plus lag/rolling/month features.
        Rows with NaN from lagging are dropped.
    """
    result = df.copy()

    # Lag features — capture autocorrelation at different horizons
    for lag in [1, 2, 3, 12]:
        result[f"lag_{lag}"] = result[target_col].shift(lag)

    # Rolling mean features — smooth out noise
    result["rolling_mean_3"] = result[target_col].shift(1).rolling(window=3).mean()
    result["rolling_mean_6"] = result[target_col].shift(1).rolling(window=6).mean()

    # Calendar feature — month number for seasonality
    result["month_number"] = result["month"].dt.month

    # Drop rows where lag features are NaN (first 12 rows)
    result = result.dropna().reset_index(drop=True)

    return result


def get_xgb_feature_cols() -> list:
    """Return the feature column names used by the XGBoost model."""
    return [
        "lag_1", "lag_2", "lag_3", "lag_12",
        "rolling_mean_3", "rolling_mean_6", "month_number",
    ]


def prepare_clustering_features(df: pd.DataFrame) -> tuple:
    """
    Prepare the feature matrix for clustering.

    Features included:
    - action_* (8 binary): type of attack action
    - actor_* (4 binary): type of threat actor
    - pattern_* (8 binary): VERIS attack pattern category
    - industry_sector (one-hot encoded, grouped by 2-digit NAICS)

    Parameters
    ----------
    df : pd.DataFrame
        The clustering dataset with all required columns.

    Returns
    -------
    tuple (X, feature_names)
        X : np.ndarray — the feature matrix
        feature_names : list — column names matching X's columns
    """
    action_cols = get_action_columns()
    actor_cols = get_actor_columns()
    pattern_cols = get_pattern_columns()

    # One-hot encode industry_sector using known NAICS codes
    # Group 31/32/33 → Manufacturing, 44/45 → Retail, 48/49 → Transportation
    sector_groups = {}
    for code, name in NAICS_MAP.items():
        sector_groups[code] = name

    # Create grouped sector column for one-hot encoding
    df_work = df.copy()
    df_work["sector_group"] = df_work["industry_sector"].map(
        lambda x: NAICS_MAP.get(str(x).zfill(2), "Other")
    )

    # One-hot encode the grouped sectors
    sector_dummies = pd.get_dummies(df_work["sector_group"], prefix="sector")

    # Combine all features into a single matrix
    feature_df = pd.concat([
        df_work[action_cols].reset_index(drop=True),
        df_work[actor_cols].reset_index(drop=True),
        df_work[pattern_cols].reset_index(drop=True),
        sector_dummies.reset_index(drop=True),
    ], axis=1)

    feature_names = list(feature_df.columns)
    X = feature_df.values.astype(float)

    return X, feature_names
