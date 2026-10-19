"""
PharmaPal-AI Clinical Dashboard - Main Application Entrypoint
Provides global patient and admission selection with clinical snapshot metrics.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

from services.clinical_queries import get_diagnoses

load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="PharmaPal-AI Clinical Dashboard",
    page_icon="💊",
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

# Fetch populated admissions directly from labevents & prescriptions_new_clean
@st.cache_data(ttl=300, show_spinner="Fetching patient admissions details...")
def fetch_valid_admissions(_eng) -> pd.DataFrame:
    query = """
        SELECT DISTINCT 
            l.subject_id, 
            l.hadm_id, 
            COUNT(DISTINCT l.itemid) AS total_labs,
            COUNT(DISTINCT p.drug) AS total_meds
        FROM labevents l
        JOIN prescriptions_new_clean p 
          ON l.subject_id = p.subject_id 
         AND l.hadm_id = p.hadm_id
        GROUP BY l.subject_id, l.hadm_id
        HAVING COUNT(DISTINCT l.itemid) > 5 
           AND COUNT(DISTINCT p.drug) > 2
        ORDER BY l.subject_id ASC, total_labs DESC
        LIMIT 100;
    """
    try:
        return pd.read_sql(query, _eng)
    except Exception:
        fallback_query = "SELECT DISTINCT subject_id, hadm_id, 0 as total_labs, 0 as total_meds FROM admissions LIMIT 20;"
        return pd.read_sql(fallback_query, _eng)

valid_admissions_df = fetch_valid_admissions(engine)

# Title & Header
st.title("💊 PharmaPal-AI: Clinical RAG Assistant")

st.header("Patient & Admission Selection")

if not valid_admissions_df.empty:
    # Normalize column names to lowercase
    valid_admissions_df.columns = [str(c).lower().strip() for c in valid_admissions_df.columns]
    valid_admissions_df["subject_id"] = valid_admissions_df["subject_id"].astype("int64")
    valid_admissions_df["hadm_id"] = valid_admissions_df["hadm_id"].astype("int64")

    # Step 1: Select Subject ID
    unique_patients = sorted(valid_admissions_df["subject_id"].unique().tolist())
    
    default_subj_idx = 0
    if "subject_id" in st.session_state and st.session_state["subject_id"] in unique_patients:
        default_subj_idx = unique_patients.index(st.session_state["subject_id"])
        
    col1, col2 = st.columns(2)
    
    with col1:
        selected_subject_id = st.selectbox(
            "Select Patient (Subject ID)", 
            options=unique_patients,
            index=default_subj_idx,
            format_func=lambda x: f"{int(x)}"
        )
    
    # Step 2: Dynamically Filter Admissions for Selected Patient
    patient_admissions = valid_admissions_df[
        valid_admissions_df["subject_id"] == selected_subject_id
    ]
    
    admission_list = sorted(patient_admissions["hadm_id"].unique().tolist())
    
    default_hadm_idx = 0
    if "hadm_id" in st.session_state and st.session_state["hadm_id"] in admission_list:
        default_hadm_idx = admission_list.index(st.session_state["hadm_id"])
        
    with col2:
        selected_hadm_id = st.selectbox(
            "Select Admission (HADM_ID)", 
            options=admission_list,
            index=default_hadm_idx,
            format_func=lambda x: f"{int(x)}",
            help="Switch between multiple admissions for this patient."
        )
    
    subject_id = int(selected_subject_id)
    hadm_id = int(selected_hadm_id)
else:
    subject_id = st.number_input("Subject ID", value=10001884, step=1)
    hadm_id = st.number_input("Admission ID (HADM_ID)", value=24746267, step=1)

# Persist selected patient/admission globally across secondary pages
st.session_state["subject_id"] = int(subject_id)
st.session_state["hadm_id"] = int(hadm_id)

st.divider()

# High-Contrast Metric Context Cards
col_p, col_a = st.columns(2)
col_p.metric("SELECTED PATIENT", f"{subject_id}")
col_a.metric("SELECTED ADMISSION", f"{hadm_id}")

st.divider()

# --------------------------------------------------------------------------
# Admission Overview Snapshot
# --------------------------------------------------------------------------
st.markdown("### 🔍 Admission Diagnoses Overview")

diagnoses_df = get_diagnoses(engine, subject_id, hadm_id)

if diagnoses_df is not None and not diagnoses_df.empty:
    total_dx = len(diagnoses_df)
    
    st.caption(f"**Total Recorded Conditions:** {total_dx} diagnoses found for this admission.")
    
    st.markdown("**Primary Highlights:**")
    for _, row in diagnoses_df.head(3).iterrows():
        st.markdown(f"- **ICD {row.get('icd_code', '')}**: {row.get('diagnosis_title', 'Unknown')}")
        
    with st.expander(f"📁 View Complete Diagnosis List ({total_dx} items)"):
        st.dataframe(
            diagnoses_df[["icd_code", "diagnosis_title"]], 
            use_container_width=True, 
            hide_index=True
        )
else:
    st.info("No primary diagnosis records found for this admission.")

st.divider()
st.caption("PharmaPal-AI | Clinical research assistant for experimental cohort analysis")