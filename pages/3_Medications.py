"""
PharmaPal-AI - Patient Medications
Displays active and historical prescriptions for the selected patient and admission.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

st.set_page_config(
    page_title="Medications - PharmaPal-AI",
    page_icon="💊",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
        .page-subtitle { color: #9CA3AF; font-size: 14px; margin-top: -8px; margin-bottom: 22px; }
        [data-testid="stMetric"] {
            background: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 12px 16px;
        }
        [data-testid="stMetricLabel"] { color: #9CA3AF !important; font-size: 0.85rem !important; }
        [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 1.35rem !important; font-weight: 600 !important; }
        div[data-testid="stExpander"] { background: transparent; border: 1px solid #374151; border-radius: 8px; }
        div[data-testid="stDataFrame"] { border: 1px solid #374151; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_db_engine():
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:sql@localhost:5432/MIMIC")
    return create_engine(database_url, pool_pre_ping=True)

@st.cache_data(ttl=300)
def fetch_patient_admissions(_engine, subject_id: int) -> list:
    query = text("""
        SELECT DISTINCT hadm_id
        FROM prescriptions_new_clean
        WHERE subject_id = :subject_id AND hadm_id IS NOT NULL
        ORDER BY hadm_id
    """)
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"subject_id": int(subject_id)})
        if not df.empty:
            return sorted(df["hadm_id"].astype(int).tolist())
    return []

@st.cache_data(ttl=300)
def get_medications(_engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    query = text("""
        SELECT drug, dose_val_rx, dose_unit_rx, route, starttime, stoptime
        FROM prescriptions_new_clean
        WHERE subject_id = :subject_id AND hadm_id = :hadm_id
        ORDER BY starttime ASC
    """)
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"subject_id": int(subject_id), "hadm_id": int(hadm_id)})
    return df

engine = get_db_engine()

st.title("💊 Patient Medications")
st.markdown('<div class="page-subtitle">Review active and completed prescriptions for the selected admission.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Retrieve Active Patient and Render Single Admission Dropdown
# --------------------------------------------------------------------------
subject_id = st.session_state.get("subject_id")

if subject_id is None:
    st.warning("Please select a patient from the main dashboard first.")
    st.stop()

subject_id = int(subject_id)
available_hadms = fetch_patient_admissions(engine, subject_id)

if not available_hadms:
    st.warning(f"No medication records found for Patient {subject_id}.")
    st.stop()

current_hadm = st.session_state.get("hadm_id")
default_index = 0
if current_hadm in available_hadms:
    default_index = available_hadms.index(int(current_hadm))

# Single Admission Selector on Main Page
hadm_id = st.selectbox(
    "Select Admission",
    options=available_hadms,
    index=default_index,
    format_func=lambda x: f"Admission {int(x)}",
    key="medication_admission_selector"
)

hadm_id = int(hadm_id)
st.session_state["hadm_id"] = hadm_id

st.divider()

col1, col2 = st.columns(2)
col1.metric("SELECTED PATIENT", f"{subject_id}")
col2.metric("SELECTED ADMISSION", f"{hadm_id}")

st.divider()

# --------------------------------------------------------------------------
# Fetch and Display Medications Data
# --------------------------------------------------------------------------
meds_df = get_medications(engine, subject_id, hadm_id)

if meds_df.empty:
    st.info("No prescription records found for this patient admission.")
    st.stop()

total_meds = len(meds_df)
unique_drugs = meds_df["drug"].nunique() if "drug" in meds_df.columns else 0

m1, m2 = st.columns(2)
m1.metric("Total Prescriptions", f"{total_meds}")
m2.metric("Unique Medications", f"{unique_drugs}")

st.divider()

st.subheader("Prescription Highlights")
for _, row in meds_df.head(3).iterrows():
    drug = row.get("drug", "Unknown Drug")
    dose = f"{row.get('dose_val_rx', '')} {row.get('dose_unit_rx', '')}".strip() or "Standard dose"
    route = row.get("route", "oral/IV")
    with st.container(border=True):
        st.markdown(f"**{drug}** was prescribed at {dose} via the {route} route.")

st.divider()

with st.expander(f"📁 View Complete Medication Records ({len(meds_df)} items)", expanded=False):
    st.dataframe(meds_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("PharmaPal-AI | Clinical research data overview")