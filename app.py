"""
PharmaPal-AI Clinical Dashboard - Main Application Entrypoint
Provides global patient and admission selection with clinical snapshot metrics.
"""

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db import get_engine, verify_db_connection
from services.clinical_queries import get_patient_admissions, get_diagnoses

load_dotenv()
fetch_valid_admissions = get_patient_admissions

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

# Render Header first to ensure immediate page loading
st.title("💊 PharmaPal-AI: Clinical RAG Assistant")
st.header("Patient & Admission Selection")

# Verify Database Connection status using db.py module
is_connected, db_status_msg = verify_db_connection()
if not is_connected:
    st.error(f"⚠️ {db_status_msg}")
    st.info("Ensure local PostgreSQL server is running and `DATABASE_URL` in `.env` is correct.")
    st.stop()

# Obtain cached database engine from db.py
engine = get_engine()

# Safe data loading with visual loader
@st.cache_data(ttl=300, show_spinner=False)
def load_admissions_data(_eng):
    try:
        return fetch_valid_admissions(_eng)
    except Exception as err:
        st.warning(f"Could not load dynamic admissions cohort: {err}")
        return pd.DataFrame()

with st.spinner("Loading patient admissions..."):
    valid_admissions_df = load_admissions_data(engine)

# Cohort Selection Logic
if valid_admissions_df is not None and not valid_admissions_df.empty:
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
    st.warning("⚠️ No cohort admissions returned from database. Falling back to default ID selection.")
    col1, col2 = st.columns(2)
    with col1:
        subject_id = st.number_input("Subject ID", value=10001884, step=1)
    with col2:
        hadm_id = st.number_input("Admission ID (HADM_ID)", value=24746267, step=1)

# Persist selected patient/admission globally across secondary pages
st.session_state["subject_id"] = int(subject_id)
st.session_state["hadm_id"] = int(hadm_id)

st.divider()

# --------------------------------------------------------------------------
# Patient Vital Status Calculation
# --------------------------------------------------------------------------
current_admission_df = pd.DataFrame()
if valid_admissions_df is not None and not valid_admissions_df.empty:
    current_admission_df = valid_admissions_df[
        (valid_admissions_df["subject_id"] == subject_id) & 
        (valid_admissions_df["hadm_id"] == hadm_id)
    ]

vital_status_label = "🟢 ALIVE"
expire_detail = "Discharged Home / Facilities"

if not current_admission_df.empty:
    row = current_admission_df.iloc[0]
    
    is_expired = (
        (row.get("hospital_expire_flag") == 1) or 
        pd.notna(row.get("dod")) or 
        pd.notna(row.get("deathtime"))
    )
    
    if is_expired:
        vital_status_label = "🔴 DECEASED"
        if row.get("hospital_expire_flag") == 1:
            deathtime_val = row.get("deathtime", "N/A")
            expire_detail = f"In-Hospital Expired: {deathtime_val}"
        elif pd.notna(row.get("dod")):
            expire_detail = f"Date of Death: {row.get('dod')}"

# --------------------------------------------------------------------------
# Dynamic Metric Context Cards (Top Bar)
# --------------------------------------------------------------------------
col_p, col_a, col_s = st.columns(3)
col_p.metric("SELECTED PATIENT", f"{subject_id}")
col_a.metric("SELECTED ADMISSION", f"{hadm_id}")
col_s.metric("PATIENT VITAL STATUS", vital_status_label, delta=expire_detail, delta_color="off")

st.divider()

# --------------------------------------------------------------------------
# Admission Overview Snapshot
# --------------------------------------------------------------------------
st.markdown("### 🔍 Admission Diagnoses Overview")

try:
    diagnoses_df = get_diagnoses(engine, subject_id, hadm_id)
except Exception as e:
    st.error(f"Error retrieving diagnoses: {e}")
    diagnoses_df = pd.DataFrame()

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