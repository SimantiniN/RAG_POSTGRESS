"""
PharmaPal-AI - Patient Diagnoses Page
Displays comprehensive ICD diagnosis codes and clinical titles for the selected admission.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Centralized service imports
from services.clinical_queries import get_diagnoses, get_patient_admissions
from services.ui_components import render_patient_admission_selectors

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Patient Diagnoses | PharmaPal-AI",
    page_icon="🩺",
    layout="wide",
)

# Professional Dark Styling
st.markdown(
    """
    <style>
        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 2rem; 
        }
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] { 
            color: #9CA3AF !important; 
            font-size: 0.85rem !important; 
        }
        [data-testid="stMetricValue"] { 
            color: #FFFFFF !important; 
            font-size: 1.35rem !important; 
            font-weight: 600 !important; 
        }
        div[data-testid="stExpander"] { 
            border: 1px solid #374151; 
            border-radius: 8px; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Database Connection Engine
@st.cache_resource
def get_db_engine():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:sql@localhost:5432/MIMIC")
    return create_engine(db_url, pool_pre_ping=True)

engine = get_db_engine()
print(engine)
# Fetch populated admissions for UI validation
@st.cache_data(ttl=300, show_spinner=False)
def fetch_admissions_cohort(_eng) -> pd.DataFrame:
    return get_patient_admissions(_eng)

valid_admissions_df = fetch_admissions_cohort(engine)

# Page Header
st.title("🩺 Patient Diagnoses")
st.markdown(
    '<div style="color: #9CA3AF; font-size: 14px; margin-top: -8px; margin-bottom: 22px;">'
    'Review recorded ICD diagnosis codes, sequences, and descriptions for the selected admission.'
    '</div>', 
    unsafe_allow_html=True
)

# --------------------------------------------------------------------------
# Single Admission Selector (Main View Control)
# --------------------------------------------------------------------------
try:
    subject_id, hadm_id = render_patient_admission_selectors(valid_admissions_df)
except Exception as exc:
    st.error("Unable to load patient admission context.")
    st.stop()

if subject_id is None or hadm_id is None:
    st.warning("Please select a valid admission to continue.")
    st.stop()

st.divider()

# High-Contrast Metric Context Cards
col1, col2 = st.columns(2)
col1.metric("SELECTED PATIENT", f"{int(subject_id)}")
col2.metric("SELECTED ADMISSION", f"{int(hadm_id)}")

st.divider()

# --------------------------------------------------------------------------
# Retrieve and Display Diagnoses
# --------------------------------------------------------------------------
st.subheader("📋 Recorded ICD Diagnoses")

try:
    diagnoses_df = get_diagnoses(engine, int(subject_id), int(hadm_id))
except Exception as exc:
    st.error(f"Failed to fetch diagnoses from the database: {exc}")
    st.stop()

if diagnoses_df is not None and not diagnoses_df.empty:
    total_dx = len(diagnoses_df)
    st.caption(f"Showing **{total_dx}** recorded diagnosis items for Admission `{hadm_id}`.")
    
    # Format and present diagnoses dataframe cleanly
    st.dataframe(
        diagnoses_df[["icd_code", "icd_version", "diagnosis_title"]],
        column_config={
            "icd_code": st.column_config.TextColumn("ICD Code"),
            "icd_version": st.column_config.NumberColumn("ICD Version", format="%d"),
            "diagnosis_title": st.column_config.TextColumn("Clinical Diagnosis Description"),
        },
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No diagnosis records found for this specific admission.")

st.divider()
st.caption("PharmaPal-AI | Clinical research assistant for experimental cohort analysis")