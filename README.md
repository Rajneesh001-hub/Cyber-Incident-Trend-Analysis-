# 🛡️ Cyber Incident Trend Analysis

**Forecasting Monthly Incident Volume & Discovering Attack Patterns Using Machine Learning**

B.Tech CSE — Semester V · Machine Learning Case Study  
Data: VERIS Community Database (VCDB) · 10,391 incidents · Jan 2010 – Jun 2026

---

## 📋 Project Overview

This project analyzes **10,391 publicly reported cyber incidents** from the [VERIS Community Database (VCDB)](https://github.com/vz-risk/VCDB) to:

1. **Forecast** monthly incident volume for the next 1–6 months using time-series models.
2. **Cluster** incidents into groups revealing recurring attack patterns by action type, actor, and industry sector.

The results are presented as an interactive **Streamlit dashboard** with a dark security-ops theme (navy + teal).

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Installation & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Train and save models offline
python train.py

# 3. Launch the dashboard
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📁 Project Structure

```
├── app.py                                      # Main entry point (Streamlit app)
├── train.py                                    # Offline model training script
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
│
├── .streamlit/
│   └── config.toml                             # Theme config (dark navy, teal accent)
│
├── pages/                                      # Page modules (routed from app.py)
│   ├── page_home.py                            # Home / Problem Statement
│   ├── page_data_quality.py                    # Data & Quality Assessment
│   ├── page_eda.py                             # Exploratory Data Analysis
│   ├── page_forecasting.py                     # Time-Series Forecasting
│   ├── page_clustering.py                      # Attack Pattern Explorer
│   └── page_evaluation.py                      # Evaluation & Limitations
│
├── utils/                                      # Shared utility modules
│   ├── data_loading.py                         # Data loading, NAICS mapping, caching
│   ├── features.py                             # Feature engineering (lags, one-hot)
│   ├── models.py                               # ML model wrappers
│   └── plotting.py                             # Plotly chart builders
│
├── models/                                     # Saved model artifacts (output of train.py)
│   ├── xgb_model_A_spike_included.joblib
│   ├── xgb_model_B_spike_capped.joblib
│   ├── kmeans_with_moveit.joblib
│   ├── kmeans_without_moveit.joblib
│   ├── feature_names_with_moveit.joblib
│   ├── feature_names_without_moveit.joblib
│   ├── forecast_results_A_spike_included.csv
│   └── forecast_results_B_spike_capped.csv
│
├── cyber_incidents_monthly_forecasting.csv     # Aggregated monthly data (198 rows)
├── cyber_incidents_clustering.csv              # Incident-level data (10,391 rows)
└── analysis.ipynb                              # Jupyter notebook with full analysis
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 **Home & Problem** | Project overview, KPI cards, problem statement, workflow |
| 📂 **Data & Quality** | Data source, variable dictionary, quality issues |
| 🔬 **Exploratory Analysis** | Yearly/monthly trends, action, actor, industry breakdowns |
| 🔮 **Forecasting** | 4 models × 2 experiments, forecast chart, error analysis |
| 🔍 **Attack Patterns** | Clustering scatter, cluster profiles, attack predictor |
| 📊 **Evaluation & Limits** | Metrics tables, honest limitations, future work |

---

## 🤖 Models

### Forecasting (Task 1)
| Model | Type |
|-------|------|
| Seasonal Naive | Baseline — repeats last year's values |
| SARIMA(1,1,1)(1,1,1,12) | Classical seasonal time-series |
| Prophet | Facebook's robust forecasting library |
| XGBoost | Gradient boosting with lag/rolling features |

Two experiments run for each model:
- **Experiment A** — spike included (raw data)
- **Experiment B** — spike capped (MOVEit outlier handled)

### Clustering (Task 2)
| Model | Type |
|-------|------|
| K-Means | Centroid-based, primary model |
| DBSCAN | Density-based, no k required |
| Agglomerative | Hierarchical clustering |

---

## 📈 Key Results

| Metric | Value |
|--------|-------|
| Best forecasting model | **SARIMA** (lowest MAE on test set) |
| Optimal clusters (k) | **5** (by silhouette analysis) |
| K-Means silhouette score | ~0.32–0.33 |
| MOVEit spike (Jun 2023) | 755 incidents — handled via dual experiments |

---

## ⚠️ Known Limitations

- Data reflects **publicly reported** incidents only — subject to reporting bias
- Small monthly counts after 2018 reduce model reliability
- The MOVEit spike is not predictable from historical patterns
- Multi-label rows (18.3%) complicate clean binary clustering

---

## 📚 References

- [VERIS Community Database (VCDB)](https://github.com/vz-risk/VCDB)
- [VERIS Framework](http://veriscommunity.net/)
- [MOVEit CVE-2023-34362](https://nvd.nist.gov/vuln/detail/CVE-2023-34362)
- [Prophet by Meta](https://facebook.github.io/prophet/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
