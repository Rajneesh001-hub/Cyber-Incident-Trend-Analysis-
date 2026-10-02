# 🛡️ Cyber Incident Trend Analysis

**Forecasting Monthly Incident Volume & Discovering Attack Patterns Using Machine Learning**

B.Tech CSE — Semester V · Machine Learning Case Study

---

## 📋 Project Overview

This project analyzes **10,391 publicly reported cyber incidents** from the [VERIS Community Database (VCDB)](https://github.com/vz-risk/VCDB) to:

1. **Forecast** monthly incident volume for the next 1–6 months using time-series models (Seasonal Naive, SARIMA, Prophet, XGBoost).
2. **Cluster** incidents into groups revealing recurring attack patterns by action type, actor, and industry sector (K-Means, DBSCAN, Agglomerative).

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or later
- pip

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Train and save models offline
python train.py

# Launch the Streamlit dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application (entry point)
├── train.py                        # Offline model training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme (dark navy, teal accent)
│
├── pages/                          # Streamlit page modules
│   ├── page_home.py                # Home / Problem Statement
│   ├── page_data_quality.py        # Data & Quality Assessment
│   ├── page_eda.py                 # Exploratory Data Analysis
│   ├── page_forecasting.py         # Time-Series Forecasting
│   ├── page_clustering.py          # Attack Pattern Explorer
│   └── page_evaluation.py          # Evaluation & Limitations
│
├── utils/                          # Utility modules
│   ├── data_loading.py             # Data loading, NAICS mapping, caching
│   ├── features.py                 # Feature engineering (lags, one-hot)
│   ├── models.py                   # ML models (forecasting + clustering)
│   └── plotting.py                 # Plotly chart builders
│
├── models/                         # Saved model artifacts (created by train.py)
│
├── cyber_incidents_monthly_forecasting.csv    # Forecasting data (198 rows)
├── cyber_incidents_clustering.csv             # Clustering data (10,391 rows)
└── analysis.ipynb                             # Jupyter notebook with full analysis
```

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Home** | Project overview, KPI cards, problem statement, workflow diagram |
| **Data & Quality** | Data source, variable dictionary, known data-quality issues |
| **EDA** | Interactive charts: yearly trends, monthly trend with spike annotation, action/actor/industry analysis |
| **Forecasting** | Model comparison (4 models × 2 experiments), forecast chart, error analysis |
| **Attack Patterns** | Clustering scatter plot, cluster profiles, interactive attack predictor |
| **Evaluation** | Metrics tables, honest limitations, future work |

## 🤖 Models Used

### Forecasting (Task 1)
- **Seasonal Naive** — baseline (repeat last year's values)
- **SARIMA(1,1,1)(1,1,1,12)** — classical seasonal time-series model
- **Prophet** — Facebook's robust forecasting library
- **XGBoost** — gradient boosting with lag/rolling features

### Clustering (Task 2)
- **K-Means** (primary) — centroid-based, tunable k
- **DBSCAN** — density-based, no k needed
- **Agglomerative** — hierarchical clustering

## 📈 Key Results

- **Best forecasting model:** SARIMA (MAE ~5-19 depending on spike handling)
- **Optimal clusters:** k=5 by silhouette analysis
- **K-Means silhouette:** ~0.32-0.33 (interpretable clusters)
- **June 2023 MOVEit outlier:** 755 incidents in one month — handled via dual experiments

## ⚠️ Known Limitations

- Data reflects publicly reported incidents only (reporting bias)
- Small monthly counts after 2018 limit model reliability
- MOVEit spike is unpredictable from historical patterns
- Multi-label rows (18.3%) complicate binary clustering

## 📚 References

- [VERIS Community Database (VCDB)](https://github.com/vz-risk/VCDB)
- [VERIS Framework](http://veriscommunity.net/)
- [MOVEit CVE-2023-34362](https://nvd.nist.gov/vuln/detail/CVE-2023-34362)
