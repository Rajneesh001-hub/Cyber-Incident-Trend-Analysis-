"""
data_loading.py — Data loading and preprocessing utilities.

Handles loading the two CSV files (forecasting and clustering),
parsing dates, enforcing correct dtypes, and applying the
NAICS industry-sector mapping for human-readable names.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# NAICS 2-digit sector code → human-readable name
# Used throughout the app for industry labels
# ─────────────────────────────────────────────────────────────
NAICS_MAP = {
    "11": "Agriculture",
    "21": "Mining",
    "22": "Utilities",
    "23": "Construction",
    "31": "Manufacturing",
    "32": "Manufacturing",
    "33": "Manufacturing",
    "42": "Wholesale Trade",
    "44": "Retail Trade",
    "45": "Retail Trade",
    "48": "Transportation",
    "49": "Transportation",
    "51": "Information",
    "52": "Finance & Insurance",
    "53": "Real Estate",
    "54": "Professional Services",
    "55": "Management",
    "56": "Administrative",
    "61": "Education",
    "62": "Healthcare",
    "71": "Arts & Entertainment",
    "72": "Accommodation & Food",
    "81": "Other Services",
    "92": "Public Administration",
    "00": "Unknown",
}


def get_data_dir() -> Path:
    """Return the directory containing the CSV data files."""
    return Path(__file__).resolve().parent.parent


@st.cache_data(ttl=3600)
def load_forecasting_data(cap_spike: bool = False) -> pd.DataFrame:
    """
    Load and preprocess the monthly forecasting CSV.

    Parameters
    ----------
    cap_spike : bool
        If True, replace the June 2023 outlier (MOVEit mass-exploitation)
        with the median of the two neighboring months (May + Jul 2023).

    Returns
    -------
    pd.DataFrame
        Columns: month (datetime index), incident_count, action_* columns.
    """
    data_dir = get_data_dir()
    df = pd.read_csv(data_dir / "cyber_incidents_monthly_forecasting.csv")

    # Parse the month column as a proper datetime
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values("month").reset_index(drop=True)

    # Handle the June 2023 MOVEit outlier if requested
    if cap_spike:
        spike_idx = df[df["month"] == "2023-06-01"].index
        if len(spike_idx) > 0:
            idx = spike_idx[0]
            # Replace with median of neighboring months (May 2023, Jul 2023)
            neighbors = []
            if idx > 0:
                neighbors.append(df.loc[idx - 1, "incident_count"])
            if idx < len(df) - 1:
                neighbors.append(df.loc[idx + 1, "incident_count"])
            median_val = int(np.median(neighbors)) if neighbors else 5
            df.loc[idx, "incident_count"] = median_val

    return df


@st.cache_data(ttl=3600)
def load_clustering_data(exclude_moveit: bool = True) -> pd.DataFrame:
    """
    Load and preprocess the clustering CSV.

    Parameters
    ----------
    exclude_moveit : bool
        If True, remove rows flagged as part of the 2023 MOVEit event.

    Returns
    -------
    pd.DataFrame
        All columns preserved, industry_sector read as string to keep
        leading zeros (e.g., '00' for Unknown).
    """
    data_dir = get_data_dir()
    df = pd.read_csv(
        data_dir / "cyber_incidents_clustering.csv",
        dtype={"industry_sector": str},  # Keep leading zeros like "00"
    )

    if exclude_moveit:
        df = df[df["is_2023_moveit_event"] == 0].copy()

    df = df.reset_index(drop=True)
    return df


def map_industry_name(sector_code: str) -> str:
    """Map a NAICS 2-digit sector code to a human-readable industry name."""
    return NAICS_MAP.get(str(sector_code).zfill(2), f"Sector {sector_code}")


def get_action_columns() -> list:
    """Return the list of action_* column names."""
    return [
        "action_Malware", "action_Hacking", "action_Social",
        "action_Physical", "action_Misuse", "action_Error",
        "action_Environmental", "action_Unknown",
    ]


def get_actor_columns() -> list:
    """Return the list of actor_* column names."""
    return [
        "actor_External", "actor_Internal", "actor_Partner", "actor_Unknown",
    ]


def get_pattern_columns() -> list:
    """Return the list of pattern_* column names."""
    return [
        "pattern_Privilege_Misuse", "pattern_Miscellaneous_Errors",
        "pattern_Lost_and_Stolen_Assets", "pattern_Basic_Web_Application_Attacks",
        "pattern_System_Intrusion", "pattern_Social_Engineering",
        "pattern_Denial_of_Service", "pattern_Everything_Else",
    ]


def get_data_quality_stats() -> dict:
    """
    Return a dictionary of data-quality statistics to display
    in the Data & Quality page.
    """
    return {
        "duplicates_removed": 205,
        "missing_month_pct": 28,
        "year_cutoff": 2010,
        "total_cleaned_records": 10391,
        "moveit_spike_month": "June 2023",
        "moveit_spike_count": 755,
        "avg_monthly_count": 37,
        "multi_label_incidents": 1906,
        "environmental_action_count": 10,
        "source": "VERIS Community Database (VCDB)",
        "source_url": "https://github.com/vz-risk/VCDB",
    }
