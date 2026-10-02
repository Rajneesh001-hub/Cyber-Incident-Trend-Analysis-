"""
page_eda.py — Exploratory Data Analysis page.

Interactive charts exploring incident trends, action-type composition,
industry distribution, actor mix, and cross-tabulations.
All charts are built with Plotly for interactivity.
"""

import streamlit as st
import pandas as pd
from utils.data_loading import load_forecasting_data, load_clustering_data, NAICS_MAP
from utils.plotting import (
    plot_incidents_per_year, plot_monthly_trend, plot_action_mix_over_time,
    plot_top_industries, plot_actor_mix, plot_action_industry_heatmap,
)


def render():
    """Render the EDA page."""

    st.markdown("""
    <h2 style="color: #00D4AA;">🔬 Exploratory Data Analysis</h2>
    <p style="color: #8892B0;">Visual exploration of patterns, trends, and anomalies in cyber incident data.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    > **What this shows:** These charts reveal the key patterns in the VCDB data — how
    > incident volume has changed over time, which action types dominate, which industries
    > are most affected, and how these dimensions interact. The June 2023 MOVEit spike
    > is annotated explicitly.
    """)

    # Load both datasets
    df_forecast = load_forecasting_data()
    df_cluster = load_clustering_data(exclude_moveit=False)

    # ── 1. Incidents per Year ──
    st.markdown("### 📊 Incidents per Year")
    st.plotly_chart(plot_incidents_per_year(df_forecast), use_container_width=True)
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #00D4AA; margin-bottom: 1rem;">
        <b style="color: #00D4AA;">💡 Key Insight:</b>
        <span style="color: #C0C8D8;">
        Yearly counts peak in 2013 (~1,205) and 2023 (~810, driven by MOVEit).
        The sharp decline after 2014 reflects reduced data collection in VCDB, not fewer real-world attacks.
        Recent years (2025–2026) show very low counts because data is still being added.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 2. Monthly Trend with Spike Annotation ──
    st.markdown("### 📈 Monthly Incident Trend")
    st.plotly_chart(plot_monthly_trend(df_forecast, annotate_spike=True), use_container_width=True)
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #FF4757; margin-bottom: 1rem;">
        <b style="color: #FF4757;">⚠️ Outlier Alert:</b>
        <span style="color: #C0C8D8;">
        June 2023 shows 755 incidents — a 20× jump caused by the MOVEit mass-exploitation event.
        A single CVE (CVE-2023-34362) in Progress MOVEit Transfer allowed the Cl0p ransomware group
        to compromise hundreds of organizations. This is handled as an outlier in the forecasting models.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 3. Action Type Mix Over Time ──
    st.markdown("### 🔀 Action Type Composition Over Time")
    st.plotly_chart(plot_action_mix_over_time(df_forecast), use_container_width=True)
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #00D4AA; margin-bottom: 1rem;">
        <b style="color: #00D4AA;">💡 Key Insight:</b>
        <span style="color: #C0C8D8;">
        Early years (2010–2014) are dominated by Error and Physical actions (lost laptops, misfiled records).
        Hacking rises to dominance from 2013 onward as web attacks become more common.
        Malware spikes in certain periods (e.g., 2016–2017 ransomware wave).
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 4. Top Industries ──
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 🏢 Top Industries")
        st.plotly_chart(plot_top_industries(df_cluster, NAICS_MAP), use_container_width=True)

    with col2:
        st.markdown("### 🎭 Actor Distribution")
        st.plotly_chart(plot_actor_mix(df_cluster), use_container_width=True)

    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #00D4AA; margin-bottom: 1rem;">
        <b style="color: #00D4AA;">💡 Key Insights:</b>
        <ul style="color: #C0C8D8; margin-bottom: 0;">
            <li><b>Healthcare (62)</b> and <b>Finance & Insurance (52)</b> are the most-reported sectors — likely due to mandatory breach disclosure laws (HIPAA, GLBA).</li>
            <li><b>Public Administration (92)</b> ranks third — government agencies have high reporting obligations.</li>
            <li><b>External actors</b> dominate (~55%), but <b>internal actors</b> are substantial (~40%) — insider threat is real.</li>
            <li>The industry distribution reflects <em>reporting requirements</em> as much as actual attack targeting.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 5. Action × Industry Heatmap ──
    st.markdown("### 🔥 Action Type × Industry Heatmap")
    st.plotly_chart(plot_action_industry_heatmap(df_cluster, NAICS_MAP), use_container_width=True)
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #00D4AA; margin-bottom: 1rem;">
        <b style="color: #00D4AA;">💡 Key Insights:</b>
        <ul style="color: #C0C8D8; margin-bottom: 0;">
            <li><b>Healthcare + Error</b> is the hottest cell — misdirected records, disposal errors, and misconfigurations are a huge problem in healthcare.</li>
            <li><b>Finance + Hacking</b> and <b>Information + Hacking</b> are prominent — these sectors face targeted external attacks.</li>
            <li><b>Physical actions</b> are concentrated in Healthcare and Retail — stolen laptops, skimming devices.</li>
            <li><b>Misuse</b> is spread across Healthcare, Public Admin, and Finance — these sectors have significant insider threats.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── 6. Summary Statistics Table ──
    st.markdown("### 📋 Summary Statistics")

    sum_cols = st.columns(3)
    with sum_cols[0]:
        st.metric("Total Incidents", f"{len(df_cluster):,}")
        st.metric("Unique Industries", df_cluster["industry_sector"].nunique())

    with sum_cols[1]:
        moveit = df_cluster["is_2023_moveit_event"].sum()
        st.metric("MOVEit Events", f"{moveit:,}")
        multi_action = (df_cluster[[c for c in df_cluster.columns if c.startswith("action_")]].sum(axis=1) > 1).sum()
        st.metric("Multi-Action Incidents", f"{multi_action:,}")

    with sum_cols[2]:
        st.metric("Year Range", f"{df_cluster['year'].min()} – {df_cluster['year'].max()}")
        st.metric("Avg Monthly Count", f"{df_forecast['incident_count'].mean():.1f}")
