# 🛡️ Cyber Incident Trend Analysis — Project Documentation

**B.Tech CSE · Semester V · Machine Learning Case Study**  
**Data:** VERIS Community Database (VCDB) · 10,391 incidents · Jan 2010 – Jun 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture & File Structure](#3-architecture--file-structure)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Task 1 — Forecasting](#6-task-1--forecasting)
7. [Task 2 — Clustering](#7-task-2--clustering)
8. [End-to-End Workflow](#8-end-to-end-workflow)
9. [Dashboard Pages](#9-dashboard-pages)
10. [Model Training (offline)](#10-model-training-offline)
11. [Key Results](#11-key-results)
12. [Known Limitations](#12-known-limitations)
13. [Running the Application](#13-running-the-application)
14. [Dependencies](#14-dependencies)

---

## 1. Project Overview

This project is a **full-stack machine learning case study** that takes raw cyber incident data from a public database, cleans and processes it, trains both forecasting and clustering models, and presents the results through an interactive Streamlit dashboard.

The two core questions it answers:

| Task | Question | Approach |
|------|----------|----------|
| **Forecasting** | How many incidents will occur next month? | Time-series models on monthly counts |
| **Clustering** | What attack patterns repeat across incidents? | Unsupervised clustering on incident features |

---

## 2. Problem Statement

Security teams plan staffing, budgets, and defenses using **incomplete views of past incidents**. Publicly reported data from the VERIS Community Database (VCDB) provides a structured record of 10,391 real-world cyber incidents after cleaning.

**Goals:**
1. Forecast monthly incident volume 1–6 months ahead using classical and ML time-series models.
2. Cluster incidents into interpretable groups that expose recurring attack patterns by action type, threat actor, and industry sector.

**Key challenge:** The June 2023 MOVEit mass-exploitation event created a spike of **755 incidents in a single month** — roughly 20× the monthly average of 37. Every model is run in two experiments to handle this:

- **Experiment A** — spike included (raw data)
- **Experiment B** — spike capped to the median of neighboring months

---

## 3. Architecture & File Structure

```
Machine Learning/
│
├── app.py                          ← Streamlit entry point, CSS, routing
├── train.py                        ← Standalone offline model trainer
├── requirements.txt                ← All Python dependencies
├── README.md                       ← Quick start and project summary
├── DOCUMENTATION.md                ← This file — full technical documentation
├── analysis.ipynb                  ← Jupyter notebook with full EDA
│
├── .streamlit/
│   └── config.toml                 ← Dark theme (navy + teal) + server config
│
├── pages/                          ← One file per dashboard page
│   ├── page_home.py                ← Home / Problem Statement
│   ├── page_data_quality.py        ← Data Quality Assessment
│   ├── page_eda.py                 ← Exploratory Data Analysis
│   ├── page_forecasting.py         ← Forecasting models & results
│   ├── page_clustering.py          ← Attack Pattern Explorer
│   └── page_evaluation.py          ← Model Evaluation & Limitations
│
├── utils/                          ← Shared business logic
│   ├── data_loading.py             ← CSV loading, caching, NAICS mapping
│   ├── features.py                 ← Feature engineering (lags, one-hot)
│   ├── models.py                   ← All ML model code
│   └── plotting.py                 ← Plotly chart builders
│
├── models/                         ← Serialized artifacts (output of train.py)
│   ├── xgb_model_A_spike_included.joblib
│   ├── xgb_model_B_spike_capped.joblib
│   ├── kmeans_with_moveit.joblib
│   ├── kmeans_without_moveit.joblib
│   ├── feature_names_with_moveit.joblib
│   ├── feature_names_without_moveit.joblib
│   ├── forecast_results_A_spike_included.csv
│   └── forecast_results_B_spike_capped.csv
│
├── cyber_incidents_monthly_forecasting.csv   ← 198 rows, monthly aggregates
└── cyber_incidents_clustering.csv            ← 10,391 rows, incident-level
```

**Architecture principle:** `app.py` is purely a router. All business logic lives in `utils/`, all rendering in `pages/`, and all model artifacts in `models/`. The Streamlit app can also train models on-the-fly using `@st.cache_data`.

---

## 4. Data Pipeline

### 4.1 Raw Data Source

- **Source:** VERIS Community Database (VCDB) — [github.com/vz-risk/VCDB](https://github.com/vz-risk/VCDB)
- **Raw records:** 10,596
- **After cleaning:** 10,391 (205 duplicates removed)

### 4.2 Two Prepared Datasets

| File | Rows | Purpose |
|------|------|---------|
| `cyber_incidents_monthly_forecasting.csv` | 198 | Monthly aggregated incident counts for time-series models |
| `cyber_incidents_clustering.csv` | 10,391 | Incident-level binary feature matrix for clustering |

### 4.3 Cleaning Steps Applied

| Issue | Resolution |
|-------|-----------|
| 205 duplicate records | Removed |
| 28% missing `month` field | Records before 2010 dropped (unreliable data) |
| `industry_sector` as int | Loaded as string to preserve leading zeros (e.g. `"00"` = Unknown) |
| June 2023 MOVEit spike (755 incidents) | Handled via dual experiments (raw vs capped) |
| Multi-label incidents (18.3%) | Kept as-is; binary flags used instead of single-label encoding |
| Environmental action (10 records) | Kept but noted as negligible |

### 4.4 Data Loading (`utils/data_loading.py`)

```
load_forecasting_data(cap_spike=False)
  └── Reads CSV → parses month → optionally replaces spike with neighbor median
  └── Returns: DataFrame[month, incident_count, action_* columns]

load_clustering_data(exclude_moveit=True)
  └── Reads CSV → optionally drops is_2023_moveit_event==1 rows
  └── Returns: DataFrame with all feature columns
```

Both functions are wrapped with `@st.cache_data(ttl=3600)` to avoid re-reading on every user interaction.

### 4.5 Industry Sector Mapping

NAICS 2-digit codes are mapped to human-readable names using `NAICS_MAP`:

```
"52" → Finance & Insurance
"62" → Healthcare
"92" → Public Administration
"51" → Information
"61" → Education
... (24 sectors total)
```

---

## 5. Feature Engineering

### 5.1 Forecasting Features (`utils/features.py`)

For XGBoost, raw monthly counts are transformed into a supervised learning problem using lag and rolling features:

| Feature | Description | Why |
|---------|-------------|-----|
| `lag_1` | Count 1 month ago | Short-term momentum |
| `lag_2` | Count 2 months ago | Short-term momentum |
| `lag_3` | Count 3 months ago | Short-term momentum |
| `lag_12` | Count same month last year | Seasonal proxy |
| `rolling_mean_3` | 3-month rolling average | Noise smoothing |
| `rolling_mean_6` | 6-month rolling average | Trend smoothing |
| `month_number` | Calendar month (1–12) | Seasonal pattern |

The first 12 rows are dropped due to NaN from lagging.

### 5.2 Clustering Features (`utils/features.py`)

Each incident is represented as a **binary feature vector** combining:

| Group | Columns | Count |
|-------|---------|-------|
| Attack actions | `action_Malware`, `action_Hacking`, `action_Social`, `action_Physical`, `action_Misuse`, `action_Error`, `action_Environmental`, `action_Unknown` | 8 |
| Threat actors | `actor_External`, `actor_Internal`, `actor_Partner`, `actor_Unknown` | 4 |
| VERIS patterns | `pattern_Privilege_Misuse`, `pattern_Miscellaneous_Errors`, `pattern_Lost_and_Stolen_Assets`, `pattern_Basic_Web_Application_Attacks`, `pattern_System_Intrusion`, `pattern_Social_Engineering`, `pattern_Denial_of_Service`, `pattern_Everything_Else` | 8 |
| Industry (one-hot) | NAICS groups mapped and one-hot encoded | ~15 |

Total: ~35 binary/dummy features per incident.

---

## 6. Task 1 — Forecasting

### 6.1 Train/Test Split

A **strict time-based split** is used — never random, to prevent data leakage:

```
Jan 2010 ──────────────── Dec 2021 │ Jan 2022 ──── Jun 2026
         TRAIN (~144 months)       │    TEST (~54 months)
```

### 6.2 Models

#### Seasonal Naive (Baseline)
- Predicts by repeating the value from the same month one year ago
- Formula: `ŷ(t) = y(t − 12)`
- Serves as the minimum bar all other models must beat

#### SARIMA(1,1,1)(1,1,1,12)
- Classical seasonal ARIMA from `statsmodels`
- Order rationale:
  - `d=1`: one differencing to remove trend
  - `p=1, q=1`: low AR/MA order to avoid overfitting on ~144 training points
  - `P=1, D=1, Q=1, s=12`: seasonal component with period 12
- `enforce_stationarity=False` and `enforce_invertibility=False` for numerical stability

#### Prophet (Meta/Facebook)
- Additive model: `y(t) = trend(t) + seasonality(t) + noise`
- `yearly_seasonality=True`, weekly/daily disabled (monthly data)
- `changepoint_prior_scale=0.05` — conservative, avoids overfitting trend changes
- Robust to the MOVEit spike due to built-in outlier handling

#### XGBoost
- Gradient boosting on the 7 lag/rolling features above
- `n_estimators=100, max_depth=4, learning_rate=0.1`
- Multi-step forecasting uses a **recursive strategy**: each prediction becomes the next step's lag feature
- Limitation: recursive predictions propagate error at each step

### 6.3 Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | `mean(|y - ŷ|)` | Average absolute error in incident count |
| RMSE | `√mean((y - ŷ)²)` | Penalizes large errors more heavily |
| MAPE | `mean(|y - ŷ| / y) × 100` | Percentage error (unstable near zero) |
| sMAPE | `mean(2|y - ŷ| / (|y| + |ŷ|)) × 100` | Symmetric, more stable |

### 6.4 Dual Experiments

Both experiments run all 4 models with the same split:

- **Experiment A:** Raw data — June 2023 = 755 incidents
- **Experiment B:** Spike capped — June 2023 replaced with `median(May 2023, Jul 2023)`

This lets users see whether the spike distorts model learning and whether capping improves test-set accuracy.

---

## 7. Task 2 — Clustering

### 7.1 Why Clustering

Incident records have multiple binary attributes (action type, actor type, attack pattern, industry). Clustering groups records with similar attribute combinations to reveal **recurring attack profiles** — e.g., "external hacking against finance" or "internal misuse in healthcare."

### 7.2 Choosing k (K-Means)

Two methods are used together:

- **Elbow method:** Plot inertia vs k (k=2..10), look for the "elbow" where marginal gain flattens
- **Silhouette score:** Measures cluster cohesion (−1 to +1, higher is better); k=5 gives the best score (~0.32–0.33)

### 7.3 Algorithms

#### K-Means (Primary)
- Centroid-based, minimizes within-cluster sum of squares
- `n_init=10` to avoid local minima
- `random_state=42` for reproducibility
- Scales to 10k records efficiently
- Produces 5 balanced, interpretable clusters

#### DBSCAN (Comparison)
- Density-based; no k required
- `eps` auto-tuned using the 90th percentile of 5-nearest-neighbor distances
- Can discover arbitrarily shaped clusters
- Limitation on this dataset: sparse binary features cause many noise points (`label=-1`)

#### Agglomerative (Comparison)
- Bottom-up hierarchical clustering
- Same k=5 for fair comparison
- Slower but produces a dendrogram interpretation

### 7.4 Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|-----------|
| Silhouette Score | Cohesion vs. separation (−1 to +1) | Closer to +1 |
| Davies-Bouldin Index | Average cluster similarity (lower is better) | Closer to 0 |

### 7.5 Visualization

- **PCA 2D projection:** Used *only* for visualization, not for clustering. The actual clustering is done on the full ~35-feature space.
- **Cluster profiles:** Auto-generated from `generate_cluster_profiles()` — each cluster gets a name, top action, top actor, top pattern, and top industries.

### 7.6 Attack Predictor

An interactive predictor in the dashboard lets users select:
- Industry sector
- Actor type (External / Internal / Partner)
- Action types (Malware, Hacking, Social, etc.)

The K-Means model then predicts the nearest cluster and displays its full profile.

---

## 8. End-to-End Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     RAW DATA (VCDB)                         │
│                     10,596 records                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    [Clean & Deduplicate]
                    • Remove 205 duplicates
                    • Drop pre-2010 rows
                    • Fix dtypes
                            │
              ┌─────────────┴─────────────┐
              │                           │
    ┌─────────▼──────────┐     ┌─────────▼──────────┐
    │  FORECASTING CSV   │     │  CLUSTERING CSV    │
    │  198 monthly rows  │     │  10,391 incidents  │
    └─────────┬──────────┘     └─────────┬──────────┘
              │                           │
    [Feature Engineering]       [Feature Engineering]
    • Lag 1, 2, 3, 12           • Binary action/actor/
    • Rolling mean 3, 6           pattern flags
    • Month number              • One-hot NAICS sectors
              │                           │
    ┌─────────▼──────────┐     ┌─────────▼──────────┐
    │  TIME-BASED SPLIT  │     │  ELBOW + SILHOUETTE │
    │  Train: ≤ Dec 2021 │     │  → k=5 optimal      │
    │  Test:  ≥ Jan 2022 │     └─────────┬──────────┘
    └─────────┬──────────┘               │
              │                 ┌─────────▼──────────┐
    ┌─────────▼──────────┐     │  CLUSTERING MODELS │
    │  FORECASTING MODELS│     │  • K-Means (k=5)   │
    │  • Seasonal Naive  │     │  • DBSCAN          │
    │  • SARIMA          │     │  • Agglomerative   │
    │  • Prophet         │     └─────────┬──────────┘
    │  • XGBoost         │               │
    └─────────┬──────────┘     [Cluster Profiles]
              │                [PCA Visualization]
    [Experiment A & B]         [Attack Predictor]
    [Metrics: MAE/RMSE/sMAPE]           │
              │                          │
              └─────────────┬────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  STREAMLIT APP     │
                  │  streamlit run     │
                  │  app.py            │
                  └─────────┬──────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
    [Home Page]   [EDA / Forecasting]  [Clustering]
    [Data Quality] [Evaluation]        [Predictor]
```

---

## 9. Dashboard Pages

### 🏠 Home & Problem
- Shield logo, project title, subtitle
- 4 KPI cards: Total Incidents, Date Range, Models Used, Best Forecast
- Problem statement with VCDB link
- Project objectives (two-column: Forecasting vs Clustering)
- Workflow diagram
- Data sources section

### 📂 Data & Quality
- Data source description and VCDB context
- Variable dictionary (column definitions)
- Data quality issues documented:
  - Missing month percentage
  - MOVEit spike explanation
  - Multi-label incident breakdown
  - Deduplication stats

### 🔬 Exploratory Analysis
- **Yearly trend** — incident count per year (bar chart)
- **Monthly trend** — time series with MOVEit spike annotation
- **Action types** — breakdown of Hacking, Malware, Social Engineering, etc.
- **Actor types** — External vs Internal vs Partner
- **Industry breakdown** — top sectors by incident count
- **Correlation heatmap** — action/actor/pattern co-occurrence

### 🔮 Forecasting
- Experiment selector (A: spike included / B: spike capped)
- Horizon selector (1–6 months)
- **Model comparison chart** — all 4 models overlaid on test data
- **Metrics table** — MAE, RMSE, MAPE, sMAPE per model
- **Future forecast** — predicted values beyond the test window
- **Error analysis** — residual plots

### 🔍 Attack Patterns
- **K selection** — elbow curve + silhouette chart
- **Cluster scatter** — PCA 2D projection coloured by cluster
- **Cluster profiles** — table with name, action, actor, pattern, industries
- **Algorithm comparison** — K-Means vs DBSCAN vs Agglomerative metrics
- **Attack Predictor** — interactive form → nearest cluster prediction

### 📊 Evaluation & Limits
- Full metrics comparison table (both experiments)
- Honest limitations section
- Future work suggestions

---

## 10. Model Training (offline)

Running `python train.py` executes the full training pipeline:

```bash
python train.py
```

**What it does:**

1. **Forecasting — Experiment A** (raw spike)
   - Loads `cyber_incidents_monthly_forecasting.csv`
   - Splits: train ≤ Dec 2021, test ≥ Jan 2022
   - Trains: Seasonal Naive, SARIMA, Prophet, XGBoost
   - Saves: `forecast_results_A_spike_included.csv`, `xgb_model_A_spike_included.joblib`

2. **Forecasting — Experiment B** (spike capped)
   - Same process with June 2023 capped to `median(neighbors)`
   - Saves: `forecast_results_B_spike_capped.csv`, `xgb_model_B_spike_capped.joblib`

3. **Clustering — without MOVEit** (`exclude_moveit=True`)
   - Elbow method k=2..10, picks best k by silhouette
   - Runs K-Means (k=5), DBSCAN (auto eps), Agglomerative (k=5)
   - Saves: `kmeans_without_moveit.joblib`, `feature_names_without_moveit.joblib`

4. **Clustering — with MOVEit** (`exclude_moveit=False`)
   - Same process including the 2023 MOVEit records
   - Saves: `kmeans_with_moveit.joblib`, `feature_names_with_moveit.joblib`

If `train.py` has not been run, the Streamlit app trains models on-the-fly via `@st.cache_data` — slower on first load but fully functional.

---

## 11. Key Results

### Forecasting

| Model | Experiment | MAE | Notes |
|-------|------------|-----|-------|
| Seasonal Naive | A (spike) | ~19 | Baseline |
| **SARIMA** | **A (spike)** | **~5** | **Best overall** |
| Prophet | A (spike) | ~8 | Robust to spike |
| XGBoost | A (spike) | ~12 | Recursive error propagation |
| SARIMA | B (capped) | ~7 | Slightly worse without spike signal |

- **SARIMA consistently wins** on MAE across both experiments
- XGBoost underperforms classical models — typical for short, noisy series
- Prophet handles the spike gracefully but doesn't match SARIMA's precision

### Clustering

| Algorithm | k | Silhouette | Davies-Bouldin |
|-----------|---|-----------|----------------|
| **K-Means** | **5** | **~0.32** | **~1.8** |
| DBSCAN | auto | ~0.18 | ~2.4 |
| Agglomerative | 5 | ~0.30 | ~1.9 |

- K-Means produces the best and most interpretable clusters
- DBSCAN labels ~15% of points as noise on this binary feature space
- 5 clusters map to interpretable attack profiles (e.g., "External Hacking – Finance", "Internal Misuse – Healthcare")

---

## 12. Known Limitations

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Reporting bias** | Counts reflect what's *reported*, not what *happened* | Explicitly noted throughout the dashboard |
| **Small monthly counts** | Average of 37/month limits model learning | Discussed in Evaluation page |
| **MOVEit spike** | Single event distorts model training | Dual-experiment approach |
| **Multi-label incidents** | 18.3% of records have multiple action types | Binary flags used; noted as limitation |
| **Recursive XGBoost forecast** | Error compounds over multi-step horizon | Documented in model description |
| **PCA for clustering viz** | 2D projection loses information | PCA only for viz; clustering on full space |
| **Short history** | ~144 training months is limited for seasonal models | Acknowledged; more data would improve SARIMA |

---

## 13. Running the Application

### Prerequisites
```
Python 3.9+
pip
```

### Steps

```bash
# 1. Navigate to project directory
cd "Desktop/Machine Learning"

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Pre-train models
python train.py

# 4. Launch the app
streamlit run app.py
```

**App opens at:** http://localhost:8501

### Streamlit Config (`.streamlit/config.toml`)

```toml
[theme]
primaryColor             = "#00D4AA"   # Teal accent
backgroundColor          = "#0A0E27"   # Deep navy
secondaryBackgroundColor = "#131842"
textColor                = "#E0E6ED"

[server]
headless = true
port     = 8501

[browser]
gatherUsageStats = false
```

---

## 14. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28 | Dashboard framework |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.24 | Numerical operations |
| `plotly` | ≥5.18 | Interactive charts |
| `scikit-learn` | ≥1.3 | Clustering, PCA, metrics |
| `statsmodels` | ≥0.14 | SARIMA |
| `prophet` | ≥1.1.4 | Prophet forecasting |
| `xgboost` | ≥2.0 | Gradient boosting |
| `joblib` | ≥1.3 | Model serialization |
| `openpyxl` | ≥3.1 | Excel export support |

---

*Documentation for the Cyber Incident Trend Analysis project — B.Tech CSE, Semester V.*
