"""
page_forecasting.py — Forecasting page.

Allows the user to select a forecast horizon (1–6 months),
toggle the June 2023 spike, and compare forecasting models.
Displays forecast charts, model comparison tables, and error analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loading import load_forecasting_data
from utils.features import create_lag_features, get_xgb_feature_cols
from utils.models import (
    compute_metrics, seasonal_naive_forecast,
    train_sarima, train_prophet, train_xgboost,
    xgboost_recursive_forecast,
)
from utils.plotting import plot_forecast, plot_actual_vs_predicted, plot_model_comparison


@st.cache_data(show_spinner=False)
def run_all_models(cap_spike: bool):
    """
    Run all four forecasting models and return their results.

    This function is cached so it only runs once per spike setting.
    The time-based split: train through Dec 2021, test from Jan 2022 onward.
    This ensures no data leakage — the test set simulates future forecasting.
    """
    df = load_forecasting_data(cap_spike=cap_spike)

    # ── Time-based split: train up to Dec 2021, test from Jan 2022 ──
    # Why Dec 2021? It gives ~12 years of training data (144 months)
    # and ~4+ years of test data to evaluate against.
    split_date = pd.Timestamp("2022-01-01")
    train_df = df[df["month"] < split_date].copy()
    test_df = df[df["month"] >= split_date].copy()

    train_series = train_df.set_index("month")["incident_count"]
    test_series = test_df.set_index("month")["incident_count"]
    test_horizon = len(test_series)

    results = []

    # ── 1. Seasonal Naive Baseline ──
    naive_preds = seasonal_naive_forecast(train_series, test_horizon, period=12)
    naive_metrics = compute_metrics(test_series.values, naive_preds)
    naive_metrics["Model"] = "Seasonal Naive"
    results.append(naive_metrics)

    # ── 2. SARIMA ──
    sarima_preds, sarima_model = train_sarima(train_series, test_horizon)
    sarima_metrics = compute_metrics(test_series.values, sarima_preds)
    sarima_metrics["Model"] = "SARIMA(1,1,1)(1,1,1,12)"
    results.append(sarima_metrics)

    # ── 3. Prophet ──
    prophet_preds, prophet_full_forecast = train_prophet(train_df, test_horizon)
    prophet_metrics = compute_metrics(test_series.values, prophet_preds)
    prophet_metrics["Model"] = "Prophet"
    results.append(prophet_metrics)

    # ── 4. XGBoost ──
    df_feat = create_lag_features(df)
    feat_cols = get_xgb_feature_cols()
    # Re-split after feature engineering (some rows dropped due to lag NaN)
    train_feat = df_feat[df_feat["month"] < split_date]
    test_feat = df_feat[df_feat["month"] >= split_date]

    if len(train_feat) > 0 and len(test_feat) > 0:
        xgb_model = train_xgboost(train_feat, train_feat["incident_count"], feat_cols)
        xgb_preds = xgb_model.predict(test_feat[feat_cols])
        xgb_preds = np.maximum(xgb_preds, 0)
        xgb_metrics = compute_metrics(test_feat["incident_count"].values, xgb_preds)
        xgb_metrics["Model"] = "XGBoost"
        results.append(xgb_metrics)
    else:
        xgb_preds = np.zeros(test_horizon)
        xgb_metrics = {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "sMAPE": np.nan, "Model": "XGBoost"}
        results.append(xgb_metrics)
        xgb_model = None

    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "MAE", "RMSE", "MAPE", "sMAPE"]]

    return {
        "train_df": train_df,
        "test_df": test_df,
        "results_df": results_df,
        "naive_preds": naive_preds,
        "sarima_preds": sarima_preds,
        "prophet_preds": prophet_preds,
        "xgb_preds": xgb_preds,
        "xgb_model": xgb_model,
        "sarima_model": sarima_model,
        "df_feat": df_feat,
    }


def render():
    """Render the Forecasting page."""

    st.markdown("""
    <h2 style="color: #00D4AA;">🔮 Time-Series Forecasting</h2>
    <p style="color: #8892B0;">Forecast monthly incident volume using four different models.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    > **What this shows:** We compare four forecasting models — Seasonal Naive (baseline),
    > SARIMA, Prophet, and XGBoost with lag features — on a strict time-based split.
    > You can toggle the June 2023 MOVEit spike and adjust the forecast horizon.
    """)

    # ── Sidebar Controls ──
    st.sidebar.markdown("### ⚙️ Forecast Controls")
    horizon = st.sidebar.slider("Forecast Horizon (months)", 1, 6, 3, key="horizon_slider")
    cap_spike = not st.sidebar.toggle("Include June 2023 Spike", value=False, key="spike_toggle")

    spike_label = "Spike Capped" if cap_spike else "Spike Included"
    st.sidebar.info(f"🔧 Mode: **{spike_label}**")

    model_choice = st.sidebar.selectbox(
        "Select Model for Forecast Chart",
        ["Seasonal Naive", "SARIMA(1,1,1)(1,1,1,12)", "Prophet", "XGBoost"],
        key="model_select",
    )

    # ── Run Models ──
    with st.spinner("🚀 Training all models... this may take a moment."):
        data = run_all_models(cap_spike=cap_spike)

    train_df = data["train_df"]
    test_df = data["test_df"]
    results_df = data["results_df"]

    # ── Model Comparison Table ──
    st.markdown(f"### 📋 Model Comparison — {spike_label}")
    st.markdown("""
    <div style="color: #8892B0; font-size: 0.85rem; margin-bottom: 0.5rem;">
    Train: Jan 2010 – Dec 2021 · Test: Jan 2022 – Jun 2026 · Metrics on test set only
    </div>
    """, unsafe_allow_html=True)

    # Highlight the best model (lowest MAE)
    best_model = results_df.loc[results_df["MAE"].idxmin(), "Model"]

    # Style the results table
    def highlight_best(row):
        if row["Model"] == best_model:
            return ["background-color: rgba(0, 212, 170, 0.15); color: #00D4AA; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        results_df.style.apply(highlight_best, axis=1).format(
            {"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}", "sMAPE": "{:.2f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown(f"""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #00D4AA; margin: 1rem 0;">
        <b style="color: #00D4AA;">🏆 Best Model:</b>
        <span style="color: #C0C8D8;">
        <b>{best_model}</b> achieves the lowest MAE on the test set.
        {"This is expected — the series is noisy, short, and driven by reporting patterns rather than a predictable signal. Simple baselines often outperform complex models in such cases." if "Naive" in best_model else "This model captures some patterns that the naive baseline misses."}
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Run Both Experiments Side by Side ──
    st.markdown("### 🔄 Experiment Comparison: Spike Included vs. Capped")

    with st.spinner("Running comparison experiment..."):
        data_with_spike = run_all_models(cap_spike=False)
        data_without_spike = run_all_models(cap_spike=True)

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("**Experiment A: Spike Included**")
        st.dataframe(data_with_spike["results_df"], use_container_width=True, hide_index=True)
    with comp_col2:
        st.markdown("**Experiment B: Spike Capped**")
        st.dataframe(data_without_spike["results_df"], use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #FFC048; margin: 1rem 0;">
        <b style="color: #FFC048;">📊 Observation:</b>
        <span style="color: #C0C8D8;">
        Including the June 2023 spike dramatically inflates MAE and RMSE for all models.
        Capping the spike gives a more realistic view of model performance on typical months.
        The spike is a single extreme event that no model could predict from historical patterns alone.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Forecast Chart ──
    st.markdown(f"### 📈 Forecast Chart — {model_choice}")

    # Select the right predictions based on user choice
    pred_map = {
        "Seasonal Naive": data["naive_preds"],
        "SARIMA(1,1,1)(1,1,1,12)": data["sarima_preds"],
        "Prophet": data["prophet_preds"],
        "XGBoost": data["xgb_preds"],
    }
    preds = pred_map[model_choice]

    # For forecasting beyond test set, retrain on all data
    df_full = load_forecasting_data(cap_spike=cap_spike)
    last_date = df_full["month"].max()
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS")

    # Generate future forecast using the full dataset
    full_series = df_full.set_index("month")["incident_count"]
    if model_choice == "Seasonal Naive":
        future_preds = seasonal_naive_forecast(full_series, horizon, period=12)
        lower, upper = None, None
    elif model_choice == "SARIMA(1,1,1)(1,1,1,12)":
        future_preds, _ = train_sarima(full_series, horizon)
        # Simple prediction interval: ±2 × std of training residuals
        std_est = full_series.std()
        lower = np.maximum(future_preds - 1.96 * std_est, 0)
        upper = future_preds + 1.96 * std_est
    elif model_choice == "Prophet":
        future_preds, _ = train_prophet(df_full, horizon)
        lower, upper = None, None
    else:  # XGBoost
        df_feat_full = create_lag_features(df_full)
        feat_cols = get_xgb_feature_cols()
        if len(df_feat_full) > 0:
            xgb_full = train_xgboost(df_feat_full, df_feat_full["incident_count"], feat_cols)
            last_row = df_feat_full.iloc[-1:]
            future_preds = xgboost_recursive_forecast(xgb_full, last_row, feat_cols, horizon)
        else:
            future_preds = np.zeros(horizon)
        lower, upper = None, None

    fig = plot_forecast(
        train_df=df_full, test_df=None,
        forecast_values=future_preds, forecast_dates=forecast_dates,
        model_name=model_choice, lower=lower, upper=upper,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Show forecast values table
    forecast_table = pd.DataFrame({
        "Month": forecast_dates.strftime("%b %Y"),
        "Forecast": [f"{v:.0f}" for v in future_preds],
    })
    if lower is not None:
        forecast_table["Lower 95%"] = [f"{v:.0f}" for v in lower]
        forecast_table["Upper 95%"] = [f"{v:.0f}" for v in upper]
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Error Analysis: Actual vs Predicted on Test Set ──
    st.markdown(f"### 📊 Error Analysis — {model_choice} on Test Set")

    test_dates = test_df["month"].values
    test_actual = test_df["incident_count"].values

    # Get test predictions for the selected model
    if model_choice == "XGBoost":
        test_pred = preds[:len(test_actual)]
    else:
        test_pred = preds[:len(test_actual)]

    min_len = min(len(test_actual), len(test_pred))
    fig_err = plot_actual_vs_predicted(
        test_dates[:min_len], test_actual[:min_len],
        test_pred[:min_len], model_choice,
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ── Model Comparison Bar Chart ──
    st.markdown("### 📋 Visual Model Comparison")
    fig_comp = plot_model_comparison(results_df)
    st.plotly_chart(fig_comp, use_container_width=True)
