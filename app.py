"""
app.py — Main Streamlit application for Cyber Incident Trend Analysis.

This is the entry point. It sets up the wide layout, sidebar navigation
with icons, global CSS styling, and routes to the six page modules.

Run with: streamlit run app.py
"""

import streamlit as st

# ── Page Configuration (must be the first Streamlit command) ──
st.set_page_config(
    page_title="Cyber Incident Trend Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global Custom CSS for the security-ops dashboard look ──
st.markdown("""
<style>
    /* Import Google Fonts — Inter for a clean, modern look */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font and background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main content area */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Sidebar base */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #06091A 0%, #0B0F2A 60%, #0D1235 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.12);
    }

    /* Nav item rows */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 2px !important;
        padding: 0 0.75rem !important;
    }

    /* Each nav label */
    [data-testid="stSidebar"] .stRadio label {
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        padding: 0.6rem 0.9rem !important;
        border-radius: 10px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #A0AABB !important;
        cursor: pointer !important;
        transition: background 0.18s, color 0.18s !important;
        border: 1px solid transparent !important;
        margin: 1px 0 !important;
        background: transparent !important;
    }

    /* Label text spans — force visible */
    [data-testid="stSidebar"] .stRadio label span {
        color: #A0AABB !important;
        font-size: 0.88rem !important;
    }

    /* Hover state */
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(0, 212, 170, 0.08) !important;
        border-color: rgba(0, 212, 170, 0.18) !important;
        color: #D0D8E8 !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover span {
        color: #D0D8E8 !important;
    }

    /* Hide the radio circle dot only */
    [data-testid="stSidebar"] .stRadio label > div:first-child {
        display: none !important;
    }

    /* Hide radio widget label ("Navigation") — must come AFTER the general label rule */
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio > label p {
        display: none !important;
        height: 0 !important;
        overflow: hidden !important;
    }

    /* Selected / active item — Streamlit adds aria-checked on the input */
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(0, 212, 170, 0.12) !important;
        border-color: rgba(0, 212, 170, 0.28) !important;
        color: #00D4AA !important;
    }

    [data-testid="stSidebar"] .stRadio label:has(input:checked) span {
        color: #00D4AA !important;
        font-weight: 600 !important;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        color: #00D4AA !important;
        font-weight: 700;
    }

    [data-testid="stMetricLabel"] {
        color: #8892B0 !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #E0E6ED;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D4AA, #00B4D8);
        color: #0A0E27;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3);
    }

    /* Selectbox and slider */
    .stSelectbox, .stSlider {
        color: #E0E6ED;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Horizontal rule */
    hr {
        border-color: rgba(0, 212, 170, 0.1) !important;
        margin: 1.5rem 0;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #0A0E27;
    }

    ::-webkit-scrollbar-thumb {
        background: #131842;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #00D4AA;
    }

    /* Hide Streamlit's default footer and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide auto-generated pages nav in sidebar */
    [data-testid="stSidebarNav"] {display: none;}

    /* Hide Deploy button only */
    .stDeployButton {display: none;}
    [data-testid="stAppDeployButton"] {display: none;}

    /* Make sidebar collapse/expand button visible */
    [data-testid="collapsedControl"] {
        color: #00D4AA !important;
        background: #131842 !important;
        border-radius: 0 8px 8px 0 !important;
        border: 1px solid rgba(0, 212, 170, 0.3) !important;
    }

    button[kind="header"] {
        color: #00D4AA !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ──
st.sidebar.markdown("""
<div style="
    padding: 1.8rem 1.5rem 1.4rem 1.5rem;
    text-align: center;
    border-bottom: 1px solid rgba(0, 212, 170, 0.1);
    margin-bottom: 0.8rem;
">
    <div style="
        width: 52px; height: 52px;
        background: rgba(0,212,170,0.08);
        border: 1px solid rgba(0,212,170,0.25);
        border-radius: 14px;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 0.9rem;
    ">🛡️</div>
    <div style="
        color: #00D4AA;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.14em;
        margin-bottom: 0.2rem;
    ">CYBER INCIDENT</div>
    <div style="
        color: #3D4870;
        font-size: 0.62rem;
        font-weight: 600;
        letter-spacing: 0.16em;
    ">TREND ANALYSIS</div>
</div>
""", unsafe_allow_html=True)

# Navigation with icons
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠  Home & Problem",
        "📂  Data & Quality",
        "🔬  Exploratory Analysis",
        "🔮  Forecasting",
        "🔍  Attack Patterns",
        "📊  Evaluation & Limits",
    ],
    label_visibility="collapsed",
)

# ── Page Router ──
# Import page modules and route based on sidebar selection
from pages import page_home, page_data_quality, page_eda, page_forecasting, page_clustering, page_evaluation

if page == "🏠  Home & Problem":
    page_home.render()
elif page == "📂  Data & Quality":
    page_data_quality.render()
elif page == "🔬  Exploratory Analysis":
    page_eda.render()
elif page == "🔮  Forecasting":
    page_forecasting.render()
elif page == "🔍  Attack Patterns":
    page_clustering.render()
elif page == "📊  Evaluation & Limits":
    page_evaluation.render()
