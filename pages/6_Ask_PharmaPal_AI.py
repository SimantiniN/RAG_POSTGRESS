
"""
PharmaPal-AI - Clinical RAG AI Assistant Interface
Provides interactive clinical synthesis and RAG-based cohort analysis.
"""

import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from db import get_engine
from services.clinical_queries import get_patient_admissions
from services.rag_service import generate_rag_answer

load_dotenv()

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Ask PharmaPal-AI (RAG)",
    page_icon="🤖",
    layout="wide",
)

# ---------------------------------------------------------
# PROFESSIONAL STYLING
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container { 
            padding-top: 1.5rem; 
            padding-bottom: 2rem; 
        }
        .page-subtitle { 
            color: #9CA3AF; 
            font-size: 14px; 
            margin-top: -8px; 
            margin-bottom: 22px; 
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
            background: transparent; 
            border: 1px solid #374151; 
            border-radius: 8px; 
        }
        div[data-testid="stDataFrame"] { 
            border: 1px solid #374151; 
            border-radius: 8px; 
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# PAGE HEADER
# ---------------------------------------------------------

st.title("🤖 Ask PharmaPal-AI (RAG)")
st.markdown(
    """
    <div class="page-subtitle">
        Analyze patient records, trajectory findings, and drug interactions using Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# DATABASE ENGINE
# ---------------------------------------------------------

engine = get_engine()

# ---------------------------------------------------------
# RETRIEVE ACTIVE PATIENT & RENDER SINGLE ADMISSION DROPDOWN
# ---------------------------------------------------------

subject_id = st.session_state.get("subject_id")

if subject_id is None:
    st.warning("Please select a patient from the main dashboard first.")
    st.stop()

subject_id = int(subject_id)

# Fetch available admissions for this patient
try:
    patient_raw_df = get_patient_admissions(engine, subject_id)
    if patient_raw_df is not None and not patient_raw_df.empty:
        patient_raw_df.columns = [str(c).lower().strip() for c in patient_raw_df.columns]
        available_hadms = sorted(
            pd.to_numeric(patient_raw_df["hadm_id"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    else:
        available_hadms = []
except Exception:
    available_hadms = []

if not available_hadms:
    current_hadm = st.session_state.get("hadm_id")
    available_hadms = [int(current_hadm)] if current_hadm else [26184834]

current_hadm = st.session_state.get("hadm_id")
default_index = 0
if current_hadm in available_hadms:
    default_index = available_hadms.index(int(current_hadm))

# Single Admission Selector
hadm_id = st.selectbox(
    "Select Admission",
    options=available_hadms,
    index=default_index,
    format_func=lambda x: f"Admission {int(x)}",
    key="rag_admission_selector"
)

hadm_id = int(hadm_id)
st.session_state["subject_id"] = subject_id
st.session_state["hadm_id"] = hadm_id

st.divider()

# Selected Patient and Admission Context Cards
col1, col2 = st.columns(2)
col1.metric("SELECTED PATIENT", f"{subject_id}")
col2.metric("SELECTED ADMISSION", f"{hadm_id}")

st.divider()

# ---------------------------------------------------------
# RESEARCH QUESTION / CLINICAL PROMPT INPUT
# ---------------------------------------------------------

st.subheader("🔍 Clinical Research Query")

default_prompt = (
    f"Summarize admission {hadm_id} for patient {subject_id}, including recorded diagnoses, "
    "relevant laboratory findings and trajectories, medication considerations, and safety considerations."
)

question = st.text_area(
    "Research Question / Clinical Prompt",
    value=default_prompt,
    height=100,
)

st.write("")

if st.button("⚡ Generate RAG Answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a valid clinical question.")
    else:
        with st.spinner("Analyzing patient context, querying database, and executing RAG pipeline..."):
            try:
                result = generate_rag_answer(engine, question, subject_id, hadm_id)
                
                st.divider()
                st.subheader("📋 Synthesis & Clinical Answer")
                
                with st.container(border=True):
                    st.markdown(result["answer"])
                
                st.divider()
                
                # Encapsulated Evidence & Performance Metrics
                with st.expander("📁 View Retrieved Evidence & Metrics", expanded=False):
                    st.markdown("**Performance & RAG Execution Metrics**")
                    st.json(result.get("metrics", {}))
                    
                    st.markdown("---")
                    st.markdown("**Reviewed Clinical Evidence Passages**")
                    evidence_rows = result.get("reviewed_evidence", [])
                    if evidence_rows:
                        st.dataframe(
                            pd.DataFrame(evidence_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No reviewed evidence matched this query.")
            
            except Exception as exc:
                st.error("Could not generate answer. Ensure Ollama is running locally and PostgreSQL is accessible.")
                st.exception(exc)

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()
st.caption("PharmaPal-AI | Clinical research assistant for experimental cohort analysis")

