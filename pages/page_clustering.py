"""
page_clustering.py — Attack Pattern Explorer page.

Runs K-Means, DBSCAN, and Agglomerative clustering on the incident data,
shows the elbow method, PCA scatter plots, cluster profile cards,
and an interactive form for predicting attack profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loading import (
    load_clustering_data, get_action_columns, get_actor_columns,
    get_pattern_columns, NAICS_MAP,
)
from utils.features import prepare_clustering_features
from utils.models import (
    run_kmeans, run_dbscan, run_agglomerative,
    compute_elbow, run_pca_2d, generate_cluster_profiles,
    predict_nearest_cluster,
)
from utils.plotting import plot_elbow, plot_cluster_scatter


@st.cache_data(show_spinner=False)
def run_clustering_pipeline(exclude_moveit: bool, n_clusters: int):
    """
    Run the full clustering pipeline: K-Means, DBSCAN, Agglomerative.
    Cached to avoid recomputation on every interaction.
    """
    df = load_clustering_data(exclude_moveit=exclude_moveit)
    X, feature_names = prepare_clustering_features(df)

    # K-Means (primary)
    km_labels, km_model, km_sil, km_db = run_kmeans(X, n_clusters)
    km_profiles = generate_cluster_profiles(df, km_labels, feature_names)

    # DBSCAN — auto-tune eps based on data characteristics
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=5)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    eps_val = float(np.percentile(distances[:, -1], 90))
    db_labels, db_model, db_sil, db_db = run_dbscan(X, eps=eps_val, min_samples=5)

    # Agglomerative
    ag_labels, ag_model, ag_sil, ag_db = run_agglomerative(X, n_clusters)
    ag_profiles = generate_cluster_profiles(df, ag_labels, feature_names)

    # PCA 2D for visualization
    X_2d = run_pca_2d(X)

    # Elbow method
    k_values, inertias, silhouettes = compute_elbow(X, range(2, 11))

    return {
        "df": df,
        "X": X,
        "X_2d": X_2d,
        "feature_names": feature_names,
        "km_labels": km_labels, "km_model": km_model,
        "km_sil": km_sil, "km_db": km_db, "km_profiles": km_profiles,
        "db_labels": db_labels, "db_sil": db_sil, "db_db": db_db,
        "ag_labels": ag_labels, "ag_sil": ag_sil, "ag_db": ag_db,
        "ag_profiles": ag_profiles,
        "k_values": k_values, "inertias": inertias, "silhouettes": silhouettes,
    }


def render():
    """Render the Attack Pattern Explorer page."""

    st.markdown("""
    <h2 style="color: #00D4AA;">🔍 Attack Pattern Explorer</h2>
    <p style="color: #8892B0;">Unsupervised clustering to discover recurring attack patterns.</p>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    > **What this shows:** We cluster incidents by action type, actor, attack pattern, and industry
    > to find groups of similar attacks. K-Means is the primary algorithm, compared against
    > DBSCAN and Agglomerative. Each cluster gets an auto-generated human-readable profile.
    """)

    # ── Sidebar Controls ──
    st.sidebar.markdown("### ⚙️ Clustering Controls")
    n_clusters = st.sidebar.slider("Number of Clusters (k)", 2, 10, 5, key="n_clusters")
    exclude_moveit = st.sidebar.toggle("Exclude MOVEit Events", value=True, key="exclude_moveit")

    # ── Run Pipeline ──
    with st.spinner("🔄 Running clustering pipeline..."):
        data = run_clustering_pipeline(exclude_moveit=exclude_moveit, n_clusters=n_clusters)

    # ── Elbow Method ──
    st.markdown("### 🔍 Choosing Optimal k — Elbow Method")
    fig_elbow = plot_elbow(data["k_values"], data["inertias"], data["silhouettes"])
    st.plotly_chart(fig_elbow, use_container_width=True)
    st.markdown("""
    <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
        border-left: 3px solid #00D4AA; margin-bottom: 1rem;">
        <b style="color: #00D4AA;">💡 How to read:</b>
        <span style="color: #C0C8D8;">
        The "elbow" is where adding more clusters gives diminishing returns (inertia curve flattens).
        The silhouette score peaks at the best k value. We recommend k=4–6 for this dataset.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Algorithm Comparison ──
    st.markdown("### 📋 Clustering Algorithm Comparison")
    n_db_clusters = len(set(data["db_labels"])) - (1 if -1 in data["db_labels"] else 0)
    n_db_noise = (data["db_labels"] == -1).sum()

    comp_df = pd.DataFrame({
        "Algorithm": ["K-Means", "DBSCAN", "Agglomerative"],
        "Clusters Found": [n_clusters, n_db_clusters, n_clusters],
        "Silhouette Score ↑": [data["km_sil"], data["db_sil"], data["ag_sil"]],
        "Davies-Bouldin ↓": [data["km_db"], data["db_db"], data["ag_db"]],
        "Notes": [
            "Primary — balanced clusters, tunable k",
            f"{n_db_noise} noise points — density-based, no k needed",
            "Hierarchical — similar to K-Means here",
        ],
    })

    # Highlight best row
    def highlight_best(row):
        if row["Algorithm"] == "K-Means":
            return ["background-color: rgba(0, 212, 170, 0.15); color: #00D4AA; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(comp_df.style.apply(highlight_best, axis=1), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── PCA Scatter Plot ──
    st.markdown("### 🗺️ Cluster Visualization (PCA 2D)")

    cluster_names = {p["cluster_id"]: p["name"] for p in data["km_profiles"]}
    fig_scatter = plot_cluster_scatter(data["X_2d"], data["km_labels"], cluster_names)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.caption("Note: PCA is used for visualization only — clustering is done on the full feature space.")

    st.markdown("---")

    # ── Cluster Profile Cards ──
    st.markdown("### 📇 Cluster Profiles (K-Means)")

    for profile in data["km_profiles"]:
        with st.expander(f"🏷️ Cluster {profile['cluster_id']}: **{profile['name']}** — {profile['size']:,} incidents", expanded=False):
            p_cols = st.columns(4)

            with p_cols[0]:
                st.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-radius: 8px; padding: 0.8rem; text-align: center;">
                    <div style="color: #8892B0; font-size: 0.75rem;">TOP ACTION</div>
                    <div style="color: #00D4AA; font-size: 1.1rem; font-weight: 700;">{profile['top_action']}</div>
                </div>
                """, unsafe_allow_html=True)

            with p_cols[1]:
                st.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-radius: 8px; padding: 0.8rem; text-align: center;">
                    <div style="color: #8892B0; font-size: 0.75rem;">TOP ACTOR</div>
                    <div style="color: #3B82F6; font-size: 1.1rem; font-weight: 700;">{profile['top_actor']}</div>
                </div>
                """, unsafe_allow_html=True)

            with p_cols[2]:
                st.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-radius: 8px; padding: 0.8rem; text-align: center;">
                    <div style="color: #8892B0; font-size: 0.75rem;">TOP PATTERN</div>
                    <div style="color: #FFC048; font-size: 1.1rem; font-weight: 700;">{profile['top_pattern']}</div>
                </div>
                """, unsafe_allow_html=True)

            with p_cols[3]:
                st.markdown(f"""
                <div style="background: rgba(19, 24, 66, 0.5); border-radius: 8px; padding: 0.8rem; text-align: center;">
                    <div style="color: #8892B0; font-size: 0.75rem;">SIZE</div>
                    <div style="color: #FF4757; font-size: 1.1rem; font-weight: 700;">{profile['size']:,}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"**Top Industries:** {', '.join(profile['top_industries'])}")

    st.markdown("---")

    # ── MOVEit Comparison ──
    st.markdown("### 🔄 With vs. Without MOVEit Events")
    if exclude_moveit:
        with st.spinner("Running comparison with MOVEit included..."):
            data_with = run_clustering_pipeline(exclude_moveit=False, n_clusters=n_clusters)
        comp_moveit = pd.DataFrame({
            "Setting": ["Without MOVEit (main)", "With MOVEit included"],
            "Silhouette": [data["km_sil"], data_with["km_sil"]],
            "Davies-Bouldin": [data["km_db"], data_with["km_db"]],
            "Incidents": [len(data["df"]), len(data_with["df"])],
        })
        st.dataframe(comp_moveit, use_container_width=True, hide_index=True)
        st.markdown("""
        <div style="background: rgba(19, 24, 66, 0.3); border-radius: 8px; padding: 0.8rem;
            border-left: 3px solid #FFC048; margin: 1rem 0;">
            <b style="color: #FFC048;">📊 Observation:</b>
            <span style="color: #C0C8D8;">
            Including the 740+ MOVEit events can distort clusters since they all share
            the same attack signature (Hacking + Malware, External actor, System Intrusion).
            Excluding them gives more diverse, informative clusters.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Toggle 'Exclude MOVEit Events' in the sidebar to see the comparison.")

    st.markdown("---")

    # ── Interactive Attack Profile Predictor ──
    st.markdown("### 🎯 Attack Profile Predictor")
    st.markdown("""
    <div style="color: #8892B0; margin-bottom: 1rem;">
    Select an industry, actor type, and action types to find the nearest attack cluster
    and get security recommendations.
    </div>
    """, unsafe_allow_html=True)

    pred_cols = st.columns(3)

    with pred_cols[0]:
        industry = st.selectbox(
            "🏢 Industry",
            sorted(set(NAICS_MAP.values())),
            key="pred_industry",
        )

    with pred_cols[1]:
        actor = st.selectbox(
            "🎭 Actor Type",
            ["External", "Internal", "Partner", "Unknown"],
            key="pred_actor",
        )

    with pred_cols[2]:
        actions = st.multiselect(
            "⚔️ Action Types",
            ["Malware", "Hacking", "Social", "Physical", "Misuse", "Error", "Environmental", "Unknown"],
            default=["Hacking"],
            key="pred_actions",
        )

    if st.button("🔍 Find Matching Cluster", key="predict_btn", use_container_width=True):
        # Build query features
        query_features = {}

        # Set action features
        for a in actions:
            query_features[f"action_{a}"] = 1

        # Set actor feature
        query_features[f"actor_{actor}"] = 1

        # Set industry feature
        query_features[f"sector_{industry}"] = 1

        # Find nearest cluster
        cluster_id = predict_nearest_cluster(
            data["km_model"], data["X"], data["km_labels"],
            query_features, data["feature_names"],
        )

        # Find the matching profile
        profile = next((p for p in data["km_profiles"] if p["cluster_id"] == cluster_id), None)

        if profile:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #131842, #1a2150);
                border: 1px solid rgba(0,212,170,0.3); border-radius: 12px; padding: 1.5rem;
                margin-top: 1rem;">
                <h3 style="color: #00D4AA; margin-top: 0;">🎯 Matched: {profile['name']}</h3>
                <div style="color: #C0C8D8; line-height: 1.8;">
                    <b>Cluster {cluster_id}</b> — {profile['size']:,} similar incidents in the database.<br>
                    <b>Dominant Action:</b> {profile['top_action']}<br>
                    <b>Dominant Actor:</b> {profile['top_actor']}<br>
                    <b>Common Pattern:</b> {profile['top_pattern']}<br>
                    <b>Affected Industries:</b> {', '.join(profile['top_industries'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Security recommendations
            recommendations = {
                "Hacking": "🛡️ Strengthen web application firewalls, enforce MFA, conduct regular penetration testing.",
                "Malware": "🦠 Update endpoint detection, maintain offline backups, train employees on phishing.",
                "Social": "📧 Deploy email filtering, conduct security awareness training, verify unusual requests.",
                "Error": "⚙️ Implement configuration management, automate deployments, add data-loss prevention controls.",
                "Physical": "🔒 Encrypt portable devices, implement physical access controls, use asset tracking.",
                "Misuse": "👁️ Deploy user behavior analytics (UEBA), enforce least-privilege access, monitor privileged accounts.",
                "Environmental": "🌊 Ensure disaster recovery plans, use redundant infrastructure, test backup generators.",
                "Unknown": "📋 Improve incident detection and classification capabilities.",
            }

            advice = recommendations.get(profile["top_action"], "Conduct a thorough security audit.")
            st.markdown(f"""
            <div style="background: rgba(255, 192, 72, 0.08); border-left: 3px solid #FFC048;
                border-radius: 8px; padding: 1rem; margin-top: 1rem;">
                <b style="color: #FFC048;">🔐 Security Recommendation:</b>
                <div style="color: #C0C8D8; margin-top: 0.3rem;">{advice}</div>
            </div>
            """, unsafe_allow_html=True)
