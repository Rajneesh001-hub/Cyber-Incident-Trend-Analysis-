"""
plotting.py — Plotly chart builders for the Streamlit dashboard.

All charts use the 'plotly_dark' template with a custom color scheme
matching the security operations dashboard theme:
- Deep navy backgrounds (#0A0E27)
- Teal/cyan accents (#00D4AA)
- Red for alerts (#FF4757)
- Clean, interactive charts with hover tooltips
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# Global chart styling constants
# ─────────────────────────────────────────────────────────────
BG_COLOR = "rgba(10, 14, 39, 0)"  # transparent to inherit Streamlit bg
GRID_COLOR = "rgba(255, 255, 255, 0.06)"
TEXT_COLOR = "#E0E6ED"
ACCENT_TEAL = "#00D4AA"
ACCENT_CYAN = "#00B4D8"
ACCENT_RED = "#FF4757"
ACCENT_YELLOW = "#FFC048"
ACCENT_PURPLE = "#A855F7"
ACCENT_BLUE = "#3B82F6"

# Color palette for multiple series/clusters
PALETTE = [
    "#00D4AA", "#3B82F6", "#FF4757", "#FFC048",
    "#A855F7", "#00B4D8", "#F97316", "#EC4899",
    "#10B981", "#6366F1", "#EF4444", "#FBBF24",
]


def _base_layout(title: str = "", height: int = 480) -> dict:
    """Standard layout settings applied to every chart."""
    return dict(
        template="plotly_dark",
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(family="Inter, Arial, sans-serif", color=TEXT_COLOR, size=12),
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR), x=0.01),
        margin=dict(l=50, r=30, t=50, b=40),
        height=height,
        xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )


# ═══════════════════════════════════════════════════════════════
# EDA CHARTS
# ═══════════════════════════════════════════════════════════════

def plot_incidents_per_year(df: pd.DataFrame) -> go.Figure:
    """Bar chart of total incidents per year."""
    yearly = df.copy()
    yearly["year"] = yearly["month"].dt.year
    yearly_counts = yearly.groupby("year")["incident_count"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_counts["year"],
        y=yearly_counts["incident_count"],
        marker=dict(
            color=yearly_counts["incident_count"],
            colorscale=[[0, "#131842"], [0.5, ACCENT_TEAL], [1, ACCENT_CYAN]],
            line=dict(width=0),
        ),
        hovertemplate="<b>%{x}</b><br>Incidents: %{y}<extra></extra>",
    ))
    fig.update_layout(**_base_layout("📊 Total Incidents per Year"))
    fig.update_layout(xaxis_title="Year", yaxis_title="Total Incidents")
    return fig


def plot_monthly_trend(df: pd.DataFrame, annotate_spike: bool = True) -> go.Figure:
    """Monthly trend line with optional annotation for the June 2023 spike."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["month"], y=df["incident_count"],
        mode="lines",
        line=dict(color=ACCENT_TEAL, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 170, 0.08)",
        name="Monthly Incidents",
        hovertemplate="<b>%{x|%b %Y}</b><br>Incidents: %{y}<extra></extra>",
    ))

    if annotate_spike:
        spike_row = df[df["month"] == "2023-06-01"]
        if not spike_row.empty:
            fig.add_annotation(
                x="2023-06-01",
                y=spike_row["incident_count"].values[0],
                text="⚠ MOVEit<br>mass-exploit<br>(755 incidents)",
                showarrow=True,
                arrowhead=2,
                arrowcolor=ACCENT_RED,
                font=dict(color=ACCENT_RED, size=11),
                bgcolor="rgba(255, 71, 87, 0.15)",
                bordercolor=ACCENT_RED,
                borderwidth=1,
                ax=-60, ay=-60,
            )

    fig.update_layout(**_base_layout("📈 Monthly Incident Trend"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Incident Count")
    return fig


def plot_action_mix_over_time(df: pd.DataFrame) -> go.Figure:
    """Stacked area chart showing the action-type composition over time."""
    action_cols = [c for c in df.columns if c.startswith("action_")]

    fig = go.Figure()
    for i, col in enumerate(action_cols):
        fig.add_trace(go.Scatter(
            x=df["month"], y=df[col],
            mode="lines",
            name=col.replace("action_", ""),
            stackgroup="one",
            line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]),
            hovertemplate=f"<b>{col.replace('action_', '')}</b><br>"
                          f"Count: %{{y}}<br>Date: %{{x|%b %Y}}<extra></extra>",
        ))

    fig.update_layout(**_base_layout("🔀 Action Type Mix Over Time"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Count")
    return fig


def plot_top_industries(df: pd.DataFrame, naics_map: dict) -> go.Figure:
    """Horizontal bar chart of top industries by incident count."""
    sector_counts = df["industry_sector"].value_counts().head(12)
    labels = [naics_map.get(str(s).zfill(2), f"Sector {s}") for s in sector_counts.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels[::-1],
        x=sector_counts.values[::-1],
        orientation="h",
        marker=dict(
            color=sector_counts.values[::-1],
            colorscale=[[0, "#131842"], [0.5, ACCENT_BLUE], [1, ACCENT_CYAN]],
        ),
        hovertemplate="<b>%{y}</b><br>Incidents: %{x}<extra></extra>",
    ))
    fig.update_layout(**_base_layout("🏢 Top Industries by Incident Count"))
    fig.update_layout(xaxis_title="Number of Incidents", yaxis_title="")
    return fig


def plot_actor_mix(df: pd.DataFrame) -> go.Figure:
    """Pie chart of actor type distribution."""
    actor_cols = [c for c in df.columns if c.startswith("actor_")]
    actor_sums = df[actor_cols].sum()
    labels = [c.replace("actor_", "") for c in actor_sums.index]

    fig = go.Figure()
    fig.add_trace(go.Pie(
        labels=labels,
        values=actor_sums.values,
        marker=dict(colors=PALETTE[:len(labels)]),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        hole=0.45,
    ))
    fig.update_layout(**_base_layout("🎭 Actor Type Distribution", height=400))
    return fig


def plot_action_industry_heatmap(df: pd.DataFrame, naics_map: dict) -> go.Figure:
    """Heatmap of action types vs. industries."""
    action_cols = [c for c in df.columns if c.startswith("action_")]

    # Map industry codes to names
    df_copy = df.copy()
    df_copy["industry"] = df_copy["industry_sector"].map(
        lambda x: naics_map.get(str(x).zfill(2), f"Sector {x}")
    )

    # Get top 10 industries
    top_industries = df_copy["industry"].value_counts().head(10).index.tolist()
    df_filtered = df_copy[df_copy["industry"].isin(top_industries)]

    # Build the pivot table
    heat_data = df_filtered.groupby("industry")[action_cols].sum()
    heat_data.columns = [c.replace("action_", "") for c in heat_data.columns]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=heat_data.values,
        x=heat_data.columns.tolist(),
        y=heat_data.index.tolist(),
        colorscale=[[0, "#0A0E27"], [0.3, "#131842"], [0.6, ACCENT_TEAL], [1, ACCENT_CYAN]],
        hovertemplate="<b>%{y}</b> — %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(**_base_layout("🔥 Action Type × Industry Heatmap"))
    fig.update_layout(xaxis_title="Action Type", yaxis_title="Industry")
    return fig


# ═══════════════════════════════════════════════════════════════
# FORECASTING CHARTS
# ═══════════════════════════════════════════════════════════════

def plot_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  forecast_values: np.ndarray, forecast_dates: pd.DatetimeIndex,
                  model_name: str, lower=None, upper=None) -> go.Figure:
    """
    Plot historical data + test data + forecast with optional prediction interval.
    """
    fig = go.Figure()

    # Historical training data
    fig.add_trace(go.Scatter(
        x=train_df["month"], y=train_df["incident_count"],
        mode="lines", name="Training Data",
        line=dict(color=ACCENT_TEAL, width=1.5),
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y}<extra></extra>",
    ))

    # Test data (actual)
    if test_df is not None and len(test_df) > 0:
        fig.add_trace(go.Scatter(
            x=test_df["month"], y=test_df["incident_count"],
            mode="lines+markers", name="Actual (Test)",
            line=dict(color=ACCENT_YELLOW, width=1.5, dash="dot"),
            marker=dict(size=4),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y}<extra></extra>",
        ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_values,
        mode="lines+markers", name=f"Forecast ({model_name})",
        line=dict(color=ACCENT_RED, width=2),
        marker=dict(size=6, symbol="diamond"),
        hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: %{y:.0f}<extra></extra>",
    ))

    # Prediction interval band
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself",
            fillcolor="rgba(255, 71, 87, 0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% Prediction Interval",
            hoverinfo="skip",
        ))

    fig.update_layout(**_base_layout(f"🔮 Forecast — {model_name}"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Incident Count")
    return fig


def plot_actual_vs_predicted(test_dates, y_true, y_pred, model_name: str) -> go.Figure:
    """Error analysis: actual vs predicted on the test set."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=test_dates, y=y_true,
        name="Actual",
        marker_color=ACCENT_TEAL,
        opacity=0.7,
        hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=test_dates, y=y_pred,
        name="Predicted",
        marker_color=ACCENT_RED,
        opacity=0.7,
        hovertemplate="<b>%{x|%b %Y}</b><br>Predicted: %{y:.0f}<extra></extra>",
    ))

    fig.update_layout(**_base_layout(f"📊 Actual vs Predicted — {model_name}"))
    fig.update_layout(barmode="group", xaxis_title="Date", yaxis_title="Incident Count")
    return fig


def plot_model_comparison(results_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing metrics across models."""
    metrics = ["MAE", "RMSE", "sMAPE"]
    fig = go.Figure()

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=results_df["Model"],
            y=results_df[metric],
            name=metric,
            marker_color=PALETTE[i],
            hovertemplate=f"<b>%{{x}}</b><br>{metric}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(**_base_layout("📋 Model Comparison"))
    fig.update_layout(barmode="group", xaxis_title="Model", yaxis_title="Metric Value")
    return fig


# ═══════════════════════════════════════════════════════════════
# CLUSTERING CHARTS
# ═══════════════════════════════════════════════════════════════

def plot_elbow(k_values, inertias, silhouettes) -> go.Figure:
    """Elbow method + silhouette score plot for choosing k."""
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=k_values, y=inertias,
        mode="lines+markers", name="Inertia (↓ better)",
        line=dict(color=ACCENT_TEAL, width=2),
        marker=dict(size=8),
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=k_values, y=silhouettes,
        mode="lines+markers", name="Silhouette Score (↑ better)",
        line=dict(color=ACCENT_YELLOW, width=2),
        marker=dict(size=8),
    ), secondary_y=True)

    layout = _base_layout("🔍 Elbow Method — Choosing Optimal k")
    fig.update_layout(**layout)
    fig.update_xaxes(title_text="Number of Clusters (k)")
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return fig


def plot_cluster_scatter(X_2d: np.ndarray, labels: np.ndarray,
                         cluster_names: dict = None) -> go.Figure:
    """2D PCA scatter plot colored by cluster assignment."""
    fig = go.Figure()

    unique_labels = sorted(set(labels))
    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        name = cluster_names.get(cl, f"Cluster {cl}") if cluster_names else f"Cluster {cl}"
        color = PALETTE[i % len(PALETTE)] if cl != -1 else "#555"

        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1],
            mode="markers",
            name=name,
            marker=dict(color=color, size=4, opacity=0.6),
            hovertemplate=f"<b>{name}</b><br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(**_base_layout("🗺️ Cluster Scatter Plot (PCA 2D)"))
    fig.update_layout(xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
    return fig
