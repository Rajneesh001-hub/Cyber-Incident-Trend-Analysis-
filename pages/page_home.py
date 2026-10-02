"""
page_home.py — Home / Problem Statement page.

Displays the project title, problem statement, objectives,
workflow diagram, and KPI cards giving a quick overview of the dataset.
"""

import streamlit as st
from utils.data_loading import load_forecasting_data, load_clustering_data


def render():
    """Render the Home / Problem Statement page."""

    # ── Hero Header ──
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0 1rem 0;">
        <h1 style="font-size: 2.2rem; background: linear-gradient(90deg, #00D4AA, #00B4D8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight: 800; margin-bottom: 0.3rem;">
            🛡️ Cyber Incident Trend Analysis
        </h1>
        <p style="color: #8892B0; font-size: 1.05rem; max-width: 800px; margin: auto;">
            Forecasting Monthly Incident Volume &amp; Discovering Attack Patterns Using Machine Learning
        </p>
        <p style="color: #555B7A; font-size: 0.85rem;">B.Tech CSE — Semester V · Machine Learning Case Study</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── KPI Cards ──
    df_forecast = load_forecasting_data()
    df_cluster = load_clustering_data(exclude_moveit=False)

    total_incidents = df_cluster.shape[0]
    date_range = f"{df_forecast['month'].min().strftime('%b %Y')} – {df_forecast['month'].max().strftime('%b %Y')}"

    kpi_cols = st.columns(4)
    kpi_data = [
        ("📋 Total Incidents", f"{total_incidents:,}", "Cleaned VCDB records"),
        ("📅 Date Range", date_range, "Monthly time series"),
        ("🤖 Models Used", "4 + 3", "Forecasting + Clustering"),
        ("🎯 Best Forecast", "SARIMA", "Lowest MAE on test set"),
    ]

    for col, (title, value, subtitle) in zip(kpi_cols, kpi_data):
        col.markdown(f"""
        <div style="background: linear-gradient(135deg, #131842, #1a2150);
            border: 1px solid rgba(0,212,170,0.15); border-radius: 12px;
            padding: 1.2rem; text-align: center; margin-bottom: 1rem;">
            <div style="color: #8892B0; font-size: 0.8rem; text-transform: uppercase;
                letter-spacing: 0.05em;">{title}</div>
            <div style="color: #00D4AA; font-size: 1.6rem; font-weight: 700;
                margin: 0.3rem 0;">{value}</div>
            <div style="color: #555B7A; font-size: 0.75rem;">{subtitle}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Problem Statement ──
    st.markdown("### 📌 Problem Statement")
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.5); border-left: 3px solid #00D4AA;
        padding: 1.2rem; border-radius: 8px; color: #C0C8D8; line-height: 1.7;">
    Security teams often plan staffing, budgets, and defenses using <b>incomplete views of past incidents</b>.
    This project analyzes publicly reported cyber incidents from the
    <a href="https://github.com/vz-risk/VCDB" style="color: #00D4AA;">VERIS Community Database (VCDB)</a>
    — <b>10,391 unique records after cleaning</b> — to:
    <ol style="margin-top: 0.8rem;">
        <li><b>Forecast</b> monthly incident volume for the next 1–6 months using time-series models.</li>
        <li><b>Cluster</b> incidents into groups that reveal recurring attack patterns by action type, actor, and industry sector.</li>
    </ol>
    <p style="color: #8892B0; font-size: 0.9rem; margin-top: 0.8rem;">
    ⚠️ The data reflects <em>publicly reported</em> incidents only, so counts are influenced by
    reporting practices as well as real attack activity. This limitation is examined explicitly throughout the analysis.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Objectives ──
    st.markdown("### 🎯 Project Objectives")
    obj_cols = st.columns(2)

    with obj_cols[0]:
        st.markdown("""
        <div style="background: rgba(19, 24, 66, 0.5); border-radius: 10px; padding: 1rem;">
            <h4 style="color: #00D4AA;">📈 Task 1 — Forecasting</h4>
            <ul style="color: #C0C8D8; line-height: 1.9;">
                <li>Compare Seasonal Naive, SARIMA, Prophet, and XGBoost</li>
                <li>Time-based train/test split (never random)</li>
                <li>Run with and without June 2023 MOVEit spike</li>
                <li>Metrics: MAE, RMSE, MAPE, sMAPE</li>
                <li>Forecast 1–6 months ahead with intervals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with obj_cols[1]:
        st.markdown("""
        <div style="background: rgba(19, 24, 66, 0.5); border-radius: 10px; padding: 1rem;">
            <h4 style="color: #00B4D8;">🔍 Task 2 — Clustering</h4>
            <ul style="color: #C0C8D8; line-height: 1.9;">
                <li>Compare K-Means, DBSCAN, Agglomerative</li>
                <li>Choose k with elbow method + silhouette score</li>
                <li>PCA for 2D visualization only</li>
                <li>Auto-generate readable cluster profiles</li>
                <li>Interactive attack pattern predictor</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Workflow Diagram ──
    st.markdown("### 🔄 Project Workflow")
    st.markdown("""
    ```
    ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌───────────────┐
    │  RAW DATA   │───▶│  CLEANING &  │───▶│  EXPLORATORY    │───▶│   MODELING   │───▶│  STREAMLIT    │
    │  (VCDB)     │    │  PREP        │    │  DATA ANALYSIS  │    │  & TRAINING  │    │  DASHBOARD    │
    │  10,596 rows│    │  • Dedup     │    │  • Trends       │    │  • SARIMA    │    │  • Forecast   │
    │             │    │  • Filter    │    │  • Outliers     │    │  • Prophet   │    │  • Clusters   │
    │             │    │    ≥ 2010    │    │  • Distributions│    │  • XGBoost   │    │  • Explorer   │
    │             │    │  • Fix dtypes│    │  • Correlations │    │  • K-Means   │    │  • Eval       │
    └─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘    └───────────────┘
                              │                                                              │
                              └──── 10,391 cleaned records ─────────────────────────────────▶┘
    ```
    """)

    # ── Data Sources ──
    st.markdown("### 📦 Data Sources")
    st.info("""
    **VERIS Community Database (VCDB)**
    - GitHub: [vz-risk/VCDB](https://github.com/vz-risk/VCDB)
    - 10,391 unique records after removing 205 duplicates
    - Two prepared CSVs: `cyber_incidents_monthly_forecasting.csv` (198 rows) and `cyber_incidents_clustering.csv` (10,391 rows)
    - Time period: January 2010 – June 2026
    """)
