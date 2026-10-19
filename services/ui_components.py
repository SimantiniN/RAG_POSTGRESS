"""
PharmaPal-AI - UI Components Module
Renders streamlined, non-duplicative admission selection controls on secondary pages.
"""

import pandas as pd
import streamlit as st


def render_patient_admission_selectors(valid_admissions_df: pd.DataFrame):
    """
    Renders ONLY the 'Select Admission' dropdown on secondary pages using the globally
    selected subject_id set by app.py. Prevents duplicate Subject ID selectors.
    """
    if valid_admissions_df is None or valid_admissions_df.empty:
        st.warning("No valid patient admissions loaded.")
        return None, None

    # Safely normalize column names
    df = valid_admissions_df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    if "subject_id" not in df.columns or "hadm_id" not in df.columns:
        st.error("Missing required subject_id or hadm_id columns.")
        return None, None

    # 1. Retrieve active subject_id set globally in st.session_state by app.py
    current_subj = st.session_state.get("subject_id")
    unique_patients = sorted(df["subject_id"].dropna().unique().tolist())

    if not unique_patients:
        st.warning("No unique patients found.")
        return None, None

    if current_subj not in unique_patients:
        current_subj = unique_patients[0]

    st.session_state["subject_id"] = int(current_subj)

    # 2. Filter available admissions for this patient
    patient_admissions = df[df["subject_id"] == current_subj]
    available_hadms = sorted(patient_admissions["hadm_id"].dropna().unique().tolist())

    if not available_hadms:
        available_hadms = [26184834]

    default_hadm_idx = 0
    current_hadm = st.session_state.get("hadm_id")
    if current_hadm in available_hadms:
        default_hadm_idx = available_hadms.index(current_hadm)

    # 3. Render ONLY the single Select Admission dropdown box on the main view
    selected_hadm_id = st.selectbox(
        "Select Admission",
        options=available_hadms,
        index=default_hadm_idx,
        format_func=lambda x: f"Admission {int(x)}"
    )

    selected_hadm_id = int(selected_hadm_id)
    st.session_state["hadm_id"] = selected_hadm_id

    return int(current_subj), selected_hadm_id