"""
page_data_quality.py — Data & Quality page.

Shows the data source, variable dictionary, and a detailed panel
of data-quality issues with specific numbers.
"""

import streamlit as st
import pandas as pd
from utils.data_loading import (
    load_forecasting_data, load_clustering_data,
    get_data_quality_stats, NAICS_MAP,
)


def render():
    """Render the Data & Quality page."""

    st.markdown("""
    <h2 style="color: #00D4AA;">📂 Data & Quality Assessment</h2>
    <p style="color: #8892B0;">Source, variable dictionary, and known data-quality issues.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── What this shows ──
    st.markdown("""
    > **What this shows:** This page documents the two datasets used in this project,
    > explains every column, and lists known data-quality issues you should be aware of
    > when interpreting the results.
    """)

    # ── Data Source ──
    st.markdown("### 🌐 Data Source")
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.5); border-radius: 10px; padding: 1.2rem;
        border: 1px solid rgba(0,212,170,0.1);">
        <h4 style="color: #00D4AA; margin-top: 0;">VERIS Community Database (VCDB)</h4>
        <ul style="color: #C0C8D8; line-height: 1.8;">
            <li><b>Repository:</b> <a href="https://github.com/vz-risk/VCDB" style="color: #00B4D8;">
                github.com/vz-risk/VCDB</a></li>
            <li><b>Description:</b> Community-maintained database of publicly reported security incidents,
                encoded using the VERIS (Vocabulary for Event Recording and Incident Sharing) schema.</li>
            <li><b>Coverage:</b> Publicly reported incidents worldwide (reporting-biased, not exhaustive).</li>
            <li><b>Records used:</b> 10,391 unique incidents after cleaning (from ~10,596 raw entries).</li>
            <li><b>Time span:</b> January 2010 — June 2026 (years before 2010 are too sparse).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── Variable Dictionary ──
    st.markdown("### 📖 Variable Dictionary")

    tab1, tab2 = st.tabs(["📈 Forecasting Dataset", "🔍 Clustering Dataset"])

    with tab1:
        st.markdown("**`cyber_incidents_monthly_forecasting.csv`** — 198 rows, 10 columns")
        forecast_dict = pd.DataFrame({
            "Column": ["month", "incident_count", "action_Malware", "action_Hacking",
                       "action_Social", "action_Physical", "action_Misuse",
                       "action_Error", "action_Environmental", "action_Unknown"],
            "Type": ["datetime", "int", "int", "int", "int", "int", "int", "int", "int", "int"],
            "Description": [
                "First day of month (YYYY-MM-01)",
                "Total incidents that month (target variable for forecasting)",
                "Count of malware-related actions that month",
                "Count of hacking-related actions",
                "Count of social engineering actions",
                "Count of physical actions (theft, tampering)",
                "Count of insider misuse actions",
                "Count of error/accidental actions",
                "Count of environmental actions (e.g., power failure) — very rare",
                "Count of unknown/unclassified actions",
            ],
        })
        st.dataframe(forecast_dict, use_container_width=True, hide_index=True)

        st.markdown("")
        st.caption("📊 Preview of the forecasting data:")
        df_f = load_forecasting_data()
        st.dataframe(df_f.head(10), use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("**`cyber_incidents_clustering.csv`** — 10,391 rows, 28 columns")
        cluster_dict = pd.DataFrame({
            "Column": [
                "incident_id", "year", "industry_sector",
                "action_* (8 cols)", "actor_* (4 cols)", "pattern_* (8 cols)",
                "data_disclosure_* (4 cols)", "is_2023_moveit_event",
            ],
            "Type": ["string", "int", "string (NAICS 2-digit)",
                     "binary (0/1)", "binary (0/1)", "binary (0/1)",
                     "binary (0/1)", "binary (0/1)"],
            "Description": [
                "Unique VCDB incident identifier (UUID)",
                "Year the incident was reported",
                "2-digit NAICS sector code (read as string to keep leading zeros)",
                "Malware, Hacking, Social, Physical, Misuse, Error, Environmental, Unknown",
                "External, Internal, Partner, Unknown — who did it",
                "Privilege Misuse, Misc Errors, Lost/Stolen Assets, Basic Web App Attacks, System Intrusion, Social Engineering, DoS, Everything Else",
                "Yes, No, Potentially, Unknown — was data disclosed",
                "1 if part of the June 2023 MOVEit mass-exploitation event",
            ],
        })
        st.dataframe(cluster_dict, use_container_width=True, hide_index=True)

        st.markdown("")
        st.caption("📊 Preview of the clustering data:")
        df_c = load_clustering_data(exclude_moveit=False)
        st.dataframe(df_c.head(10), use_container_width=True, hide_index=True)

    # ── NAICS Industry Mapping ──
    st.markdown("### 🏢 NAICS Industry Sector Mapping")
    naics_df = pd.DataFrame([
        {"Code": code, "Industry": name}
        for code, name in sorted(NAICS_MAP.items())
    ])
    # Remove duplicate industry names (31/32/33 all map to Manufacturing)
    naics_df = naics_df.drop_duplicates(subset="Industry")
    st.dataframe(naics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Data Quality Issues Panel ──
    st.markdown("### ⚠️ Known Data-Quality Issues")
    stats = get_data_quality_stats()

    # Display issues as styled cards
    issues = [
        ("🔄 Duplicates Removed", f"{stats['duplicates_removed']} duplicate incident IDs were removed during cleaning.",
         "#FF4757"),
        ("📅 Missing Months", f"{stats['missing_month_pct']}% of raw VCDB rows had no month field and were assigned to January of their year.",
         "#FFC048"),
        ("📉 Sparse Early Years", "Years before 2010 have very few entries — the 2010 cutoff was chosen because earlier data is unreliable.",
         "#FFC048"),
        ("📊 Uneven Yearly Counts", "Yearly counts vary wildly (e.g., 2013 had 1,205 incidents vs. 2018 with 247). This reflects collection/reporting bias, NOT real trends.",
         "#FF4757"),
        ("💥 June 2023 Outlier (MOVEit)", f"June 2023 has {stats['moveit_spike_count']} incidents (avg. is ~{stats['avg_monthly_count']}). This is the MOVEit mass-exploitation event — a single vulnerability affecting hundreds of organizations simultaneously.",
         "#FF4757"),
        ("🏷️ Multi-Label Rows", f"{stats['multi_label_incidents']:,} incidents (18.3%) have more than one action type marked. These are genuine multi-vector attacks, not errors.",
         "#FFC048"),
        ("⚖️ Class Imbalance", f"Environmental action has only {stats['environmental_action_count']} rows out of 10,391 — extreme class imbalance.",
         "#FFC048"),
        ("📰 Reporting Bias", "VCDB contains only publicly reported incidents. The true number of cyber incidents is much higher. Counts reflect reporting practices, not actual attack activity.",
         "#FF4757"),
    ]

    for i in range(0, len(issues), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(issues):
                title, desc, color = issues[i + j]
                col.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-left: 3px solid {color};
                    border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; min-height: 130px;">
                    <div style="color: {color}; font-weight: 600; margin-bottom: 0.4rem;">{title}</div>
                    <div style="color: #C0C8D8; font-size: 0.9rem; line-height: 1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
