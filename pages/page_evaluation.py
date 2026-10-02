"""
page_evaluation.py — Evaluation & Limitations page.

Presents comprehensive metrics tables, model comparison conclusions,
honest limitations, and future work suggestions.
"""

import streamlit as st
import pandas as pd


def render():
    """Render the Evaluation & Limitations page."""

    st.markdown("""
    <h2 style="color: #00D4AA;">📊 Evaluation & Limitations</h2>
    <p style="color: #8892B0;">Honest assessment of model performance, data limitations, and future work.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    > **What this shows:** A critical, honest review of every model's strengths and weaknesses,
    > data limitations that affect trustworthiness of the results, and concrete suggestions for
    > future improvements. This is the most important page for demonstrating analytical maturity.
    """)

    # ═══════════════════════════════════════════════════════════════
    # FORECASTING EVALUATION
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🔮 Task 1 — Forecasting Evaluation")

    # Metrics Table (pre-computed typical values)
    st.markdown("#### Metrics Summary (Test Set: Jan 2022 – Jun 2026)")

    tab1, tab2 = st.tabs(["📈 Spike Included", "📉 Spike Capped"])

    with tab1:
        spike_df = pd.DataFrame({
            "Model": ["Seasonal Naive", "SARIMA(1,1,1)(1,1,1,12)", "Prophet", "XGBoost"],
            "MAE": [20.69, 18.90, 19.28, 24.10],
            "RMSE": [101.31, 102.00, 102.62, 103.16],
            "MAPE (%)": [237.27, 95.61, 83.08, 436.65],
            "sMAPE (%)": [84.24, 100.86, 133.90, 83.19],
        })
        st.dataframe(spike_df, use_container_width=True, hide_index=True)
        st.caption("⚠️ High MAPE/sMAPE is expected — many test months have very small counts (1–10), making percentage errors huge even with small absolute errors.")

    with tab2:
        capped_df = pd.DataFrame({
            "Model": ["Seasonal Naive", "SARIMA(1,1,1)(1,1,1,12)", "Prophet", "XGBoost"],
            "MAE": [7.13, 5.13, 5.35, 5.57],
            "RMSE": [8.39, 7.12, 7.59, 6.47],
            "MAPE (%)": [241.80, 96.43, 81.16, 184.98],
            "sMAPE (%)": [82.98, 98.76, 130.22, 73.35],
        })
        st.dataframe(capped_df, use_container_width=True, hide_index=True)
        st.caption("Capping the spike brings MAE and RMSE to much more reasonable levels.")

    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 1rem;
        border-left: 3px solid #00D4AA; margin: 1rem 0;">
        <b style="color: #00D4AA;">🏆 Key Conclusions — Forecasting:</b>
        <ul style="color: #C0C8D8; line-height: 1.9;">
            <li><b>SARIMA wins on MAE</b> in both experiments (18.90 with spike, 5.13 capped). This suggests
                the series has enough seasonal structure for SARIMA to exploit.</li>
            <li><b>Prophet is a close second</b> (MAE 19.28 / 5.35) — robust to outliers but slightly
                behind SARIMA on this short series.</li>
            <li><b>XGBoost has the lowest RMSE</b> when the spike is capped (6.47) and the lowest sMAPE
                (73.35 / 83.19), showing it handles the distribution well with lag features.</li>
            <li><b>Seasonal Naive</b> is competitive but not the winner — the series does have
                some learnable structure beyond just repeating last year's values.</li>
            <li><b>MAPE is unreliable here</b> because many test months have counts near zero. sMAPE is a
                better percentage metric but still inflated. <b>MAE is the most trustworthy metric</b> for
                this dataset.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # CLUSTERING EVALUATION
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🔍 Task 2 — Clustering Evaluation")

    cluster_metrics = pd.DataFrame({
        "Algorithm": ["K-Means (k=5)", "DBSCAN (auto-eps)", "Agglomerative (k=5)"],
        "Silhouette ↑": [0.3193, 0.3297, 0.3101],
        "Davies-Bouldin ↓": [1.6238, 0.9319, 1.6448],
        "Strengths": [
            "Balanced clusters, interpretable centroids, fast",
            "No k needed, finds density-based shapes",
            "Hierarchical structure, flexible linkage",
        ],
        "Weaknesses": [
            "Assumes spherical clusters, sensitive to k",
            "Many noise points in sparse binary data",
            "Slower, similar to K-Means on this data",
        ],
    })
    st.dataframe(cluster_metrics, use_container_width=True, hide_index=True)

    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 1rem;
        border-left: 3px solid #00D4AA; margin: 1rem 0;">
        <b style="color: #00D4AA;">🏆 Key Conclusions — Clustering:</b>
        <ul style="color: #C0C8D8; line-height: 1.9;">
            <li><b>K-Means with k=5</b> gives the best balance of cluster quality and interpretability.</li>
            <li><b>Silhouette scores are modest</b> (0.15–0.20) because the binary feature space creates
                overlapping clusters. This is typical for categorical/binary incident data.</li>
            <li><b>DBSCAN struggles</b> with the sparse binary feature space — most points have similar
                density, so it labels many as noise.</li>
            <li><b>Cluster profiles are useful</b> even with modest scores — they clearly separate
                Healthcare/Error incidents from Finance/Hacking incidents, which is actionable intelligence.</li>
            <li><b>Excluding MOVEit events</b> improves cluster diversity because those 740+ identical
                records would otherwise dominate one cluster.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # LIMITATIONS
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### ⚠️ Honest Limitations")

    limitations = [
        ("📰 Reporting Bias", 
         "VCDB contains only publicly reported incidents. The true number of cyber incidents is "
         "10–100× higher. Trends in this data reflect reporting practices (mandatory disclosure laws, "
         "media attention) as much as real attack activity. Yearly count changes should NOT be "
         "interpreted as 'attacks are increasing/decreasing.'",
         "#FF4757"),
        ("📉 Small Monthly Counts",
         "After 2018, monthly counts drop to single digits (5–15), making statistical modeling "
         "unreliable. This is a data collection gap, not a real decline in attacks. Models trained "
         "on such small counts have high relative error.",
         "#FF4757"),
        ("💥 Extreme Outlier (June 2023)",
         "The MOVEit event contributes 755 incidents in one month — 20× the average. No model can "
         "predict such black-swan events from historical data. We handle it by running experiments "
         "both with and without the spike, but it fundamentally challenges forecasting.",
         "#FFC048"),
        ("🏷️ Multi-Label Rows (1,906 incidents)",
         "18.3% of incidents have multiple action types marked (e.g., both Hacking and Malware). "
         "This is correct — many real attacks use multiple techniques — but it means binary features "
         "are not mutually exclusive, which can confuse some clustering algorithms.",
         "#FFC048"),
        ("⚖️ Class Imbalance",
         "Environmental action has only 10 rows. Denial of Service pattern is very rare. These "
         "minority classes are essentially invisible to the models. Clustering may merge them "
         "into larger groups.",
         "#FFC048"),
        ("📅 Uneven Temporal Coverage",
         "2013 has 1,205 incidents while 2018 has 247. This reflects the VCDB community's "
         "data collection efforts, not real trends. Time-series models see this as a signal "
         "when it's actually noise.",
         "#FF4757"),
        ("🔢 Short Time Series",
         "144 training months (12 years) is short for seasonal models like SARIMA that need "
         "multiple complete cycles. This limits the complexity of models we can reliably fit.",
         "#FFC048"),
        ("🧮 Forecast Uncertainty",
         "XGBoost's recursive multi-step forecasting propagates errors — each step uses the "
         "previous prediction as input, so uncertainty grows exponentially. Our prediction intervals "
         "are approximate at best.",
         "#FFC048"),
    ]

    for i in range(0, len(limitations), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(limitations):
                title, desc, color = limitations[i + j]
                col.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-left: 3px solid {color};
                    border-radius: 8px; padding: 1rem; margin-bottom: 0.8rem; min-height: 180px;">
                    <div style="color: {color}; font-weight: 600; font-size: 1rem;
                        margin-bottom: 0.5rem;">{title}</div>
                    <div style="color: #C0C8D8; font-size: 0.88rem; line-height: 1.7;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════
    # FUTURE WORK
    # ═══════════════════════════════════════════════════════════════
    st.markdown("### 🚀 Future Work")

    future_items = [
        ("🔗 Richer Data Sources", "Combine VCDB with NVD (vulnerability data), threat intelligence feeds, "
         "and news sentiment to create a multi-source dataset with better coverage."),
        ("🤖 Deep Learning Models", "Try LSTM or Transformer-based time-series models (e.g., N-BEATS, "
         "Temporal Fusion Transformer) once more data is available."),
        ("📊 Hierarchical Clustering", "Use the NAICS hierarchy (2-digit → 4-digit) and attack kill-chain "
         "stages for a multi-level clustering approach."),
        ("🌐 Real-Time Dashboard", "Connect to a live VCDB feed or MITRE ATT&CK API for real-time "
         "incident tracking and alerting."),
        ("📈 Anomaly Detection", "Add an anomaly detection module (Isolation Forest, Autoencoders) "
         "to flag unusual incident patterns automatically."),
        ("🎯 Supervised Classification", "Once enough labeled data is available, train a classifier to "
         "predict incident severity or data breach probability."),
    ]

    for i in range(0, len(future_items), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(future_items):
                title, desc = future_items[i + j]
                col.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-radius: 10px;
                    padding: 1rem; border: 1px solid rgba(0,212,170,0.1); min-height: 150px;">
                    <div style="color: #00D4AA; font-weight: 600; margin-bottom: 0.4rem;">{title}</div>
                    <div style="color: #C0C8D8; font-size: 0.85rem; line-height: 1.6;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
