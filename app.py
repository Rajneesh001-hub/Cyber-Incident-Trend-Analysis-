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

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080C20 0%, #0D1235 100%);
        border-right: 1px solid rgba(0, 212, 170, 0.08);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* Sidebar radio buttons — navigation items */
    [data-testid="stSidebar"] .stRadio > label {
        color: #8892B0 !important;
        font-size: 0.95rem;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(0, 212, 170, 0.08);
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

    /* Hide Deploy button in toolbar */
    [data-testid="stToolbar"] {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stAppDeployButton"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ── Sidebar Navigation ──
st.sidebar.markdown("""
<div style="text-align: center; padding: 0.5rem 0 1.5rem 0;">
    <div style="font-size: 2rem; margin-bottom: 0.2rem;">🛡️</div>
    <div style="color: #00D4AA; font-size: 1rem; font-weight: 700; letter-spacing: 0.03em;">
        CYBER INCIDENT
    </div>
    <div style="color: #555B7A; font-size: 0.75rem; letter-spacing: 0.08em;">
        TREND ANALYSIS
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation with icons
page = st.sidebar.radio(
    "📍 Navigation",
    [
        "🏠 Home & Problem",
        "📂 Data & Quality",
        "🔬 Exploratory Analysis",
        "🔮 Forecasting",
        "🔍 Attack Patterns",
        "📊 Evaluation & Limits",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="color: #555B7A; font-size: 0.7rem; text-align: center; padding: 0.5rem;">
    B.Tech CSE · Semester V<br>
    Machine Learning Case Study<br>
    Data: VCDB · 10,391 incidents
</div>
""", unsafe_allow_html=True)

# ── Page Router ──
# Import page modules and route based on sidebar selection
from pages import page_home, page_data_quality, page_eda, page_forecasting, page_clustering, page_evaluation

if page == "🏠 Home & Problem":
    page_home.render()
elif page == "📂 Data & Quality":
    page_data_quality.render()
elif page == "🔬 Exploratory Analysis":
    page_eda.render()
elif page == "🔮 Forecasting":
    page_forecasting.render()
elif page == "🔍 Attack Patterns":
    page_clustering.render()
elif page == "📊 Evaluation & Limits":
    page_evaluation.render()
