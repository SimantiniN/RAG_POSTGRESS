"""
PharmaPal-AI - Laboratory Results
Displays lab events and test measurements recorded during admission.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

st.set_page_config(
    page_title="Laboratory Results - PharmaPal-AI",
    page_icon="🩸",
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
        FROM labevents
        WHERE subject_id = :subject_id AND hadm_id IS NOT NULL
        ORDER BY hadm_id
    """)
    with _engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"subject_id": int(subject_id)})
        if not df.empty:
            return sorted(df["hadm_id"].astype(int).tolist())
    return []

@st.cache_data(ttl=300)
def get_labs(_engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    query = text("""
        SELECT itemid, valuenum, valueuom, flag, charttime
        FROM labevents
        WHERE subject_id = :subject_id AND hadm_id = :hadm_id
        ORDER BY charttime DESC
    """)
    with _engine.connect() as conn:
        return pd.read_sql(query, conn, params={"subject_id": int(subject_id), "hadm_id": int(hadm_id)})

engine = get_db_engine()

st.title("🩸 Laboratory Results")
st.markdown('<div class="page-subtitle">Inspect charted laboratory events and recorded clinical flags.</div>', unsafe_allow_html=True)

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
    st.warning(f"No admission records found for Patient {subject_id}.")
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
    key="lab_admission_selector"
)

hadm_id = int(hadm_id)
st.session_state["hadm_id"] = hadm_id

st.divider()

# Selected Patient and Admission Cards
col1, col2 = st.columns(2)
col1.metric("SELECTED PATIENT", str(subject_id))
col2.metric("SELECTED ADMISSION", str(hadm_id))

st.divider()

# --------------------------------------------------------------------------
# Fetch and Display Laboratory Data
# --------------------------------------------------------------------------
labs_df = get_labs(engine, subject_id, hadm_id)

if labs_df.empty:
    st.info("No laboratory records found for this patient admission.")
    st.stop()

abnormal_count = len(labs_df[labs_df["flag"].str.lower() == "abnormal"]) if "flag" in labs_df.columns else 0

m1, m2 = st.columns(2)
m1.metric("Total Recorded Lab Tests", f"{len(labs_df)}")
m2.metric("Abnormal Lab Flags", f"{abnormal_count}")

st.divider()

st.subheader("Laboratory Highlights")
for _, row in labs_df.head(3).iterrows():
    val = f"{row.get('valuenum', 'N/A')} {row.get('valueuom', '')}".strip()
    flag = row.get("flag", "normal")
    with st.container(border=True):
        st.markdown(f"**Item ID {row.get('itemid')}** measured {val} with flag status: {flag}.")

st.divider()

with st.expander(f"📁 View Complete Laboratory Events ({len(labs_df)} records)", expanded=False):
    st.dataframe(labs_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("PharmaPal-AI | Clinical research data overview")