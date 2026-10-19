
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

from services.clinical_queries import (
    get_patient_profile,
    get_diagnoses,
)
from services.ui_components import render_patient_admission_selectors


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

load_dotenv()

st.set_page_config(
    page_title="Patient Overview | PharmaPal-AI",
    page_icon="📋",
    layout="wide",
)


# =========================================================
# 2. PAGE STYLING
# =========================================================

st.markdown(
    """
    <style>
        div[data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.06);
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 14px 16px;
            border-radius: 10px;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.88rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }

        div[data-testid="stExpander"] {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. DATABASE CONNECTION
# =========================================================

@st.cache_resource
def get_db_engine():
    """Create and reuse the PostgreSQL database engine."""

    database_url = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:sql@localhost:5432/MIMIC",
    )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


# =========================================================
# 4. FETCH ADMISSIONS
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_valid_admissions(_engine):
    """
    Fetch admissions with sufficient laboratory and medication
    data. Fall back to the admissions table if necessary.
    """

    query = """
        SELECT
            l.subject_id,
            l.hadm_id,
            COUNT(DISTINCT l.itemid) AS total_labs,
            COUNT(DISTINCT p.drug) AS total_meds
        FROM labevents AS l
        INNER JOIN prescriptions_new_clean AS p
            ON l.subject_id = p.subject_id
            AND l.hadm_id = p.hadm_id
        GROUP BY
            l.subject_id,
            l.hadm_id
        HAVING
            COUNT(DISTINCT l.itemid) > 5
            AND COUNT(DISTINCT p.drug) > 2
        ORDER BY
            l.subject_id ASC,
            total_labs DESC
        LIMIT 100;
    """

    try:
        admissions_df = pd.read_sql_query(query, _engine)

        if not admissions_df.empty:
            return admissions_df

    except Exception as exc:
        print(f"Filtered admissions query failed: {exc}")

    fallback_query = """
        SELECT
            subject_id,
            hadm_id,
            0 AS total_labs,
            0 AS total_meds
        FROM admissions
        ORDER BY subject_id, hadm_id
        LIMIT 100;
    """

    try:
        return pd.read_sql_query(fallback_query, _engine)

    except Exception as exc:
        print(f"Fallback admissions query failed: {exc}")

        return pd.DataFrame(
            columns=[
                "subject_id",
                "hadm_id",
                "total_labs",
                "total_meds",
            ]
        )


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def safe_display(value, default="N/A"):
    """Safely format missing or invalid values."""

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if value.lower() in {"", "nan", "none", "nat"}:
        return default

    return value


def format_age(value):
    """Format patient age."""

    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        age = float(value)

        if age.is_integer():
            return f"{int(age)} yrs"

        return f"{age:.1f} yrs"

    except (TypeError, ValueError):
        return safe_display(value)


def get_first_available(record, columns, default="N/A"):
    """Return the first populated value from available columns."""

    for column in columns:
        if column in record.index:
            value = safe_display(record[column], default)

            if value != default:
                return value

    return default


def get_diagnosis_description(record):
    """Find the best available diagnosis description."""

    return get_first_available(
        record,
        [
            "diagnosis_title",
            "long_title",
            "label",
            "description",
            "short_title",
            "title",
            "diagnosis_description",
        ],
    )


def get_diagnosis_code(record):
    """Find the ICD code from available columns."""

    return get_first_available(
        record,
        [
            "icd_code",
            "icd9_code",
            "icd10_code",
            "code",
        ],
    )


# =========================================================
# 6. PAGE HEADER
# =========================================================

st.title("📋 Patient Overview")

st.caption(
    "Explore patient demographics, admission information, "
    "diagnosis highlights, and admission history."
)

st.divider()


# =========================================================
# 7. LOAD DATABASE AND ADMISSIONS
# =========================================================

try:
    engine = get_db_engine()
    valid_admissions_df = fetch_valid_admissions(engine)

except Exception as exc:
    st.error(
        "Unable to connect to the database. "
        "Please check your DATABASE_URL configuration."
    )
    print(f"Database connection failed: {exc}")
    st.stop()


if valid_admissions_df.empty:
    st.warning("No admissions are available.")
    st.stop()


for column in ["subject_id", "hadm_id"]:
    valid_admissions_df[column] = pd.to_numeric(
        valid_admissions_df[column],
        errors="coerce",
    )

valid_admissions_df = valid_admissions_df.dropna(
    subset=["subject_id", "hadm_id"]
).copy()

if valid_admissions_df.empty:
    st.warning("No valid patient or admission IDs were found.")
    st.stop()


valid_admissions_df["subject_id"] = (
    valid_admissions_df["subject_id"].astype("int64")
)

valid_admissions_df["hadm_id"] = (
    valid_admissions_df["hadm_id"].astype("int64")
)

valid_admissions_df = valid_admissions_df.drop_duplicates(
    subset=["subject_id", "hadm_id"]
)


# =========================================================
# 8. SINGLE ADMISSION SELECTION FOR ACTIVE PATIENT
# =========================================================

# Fetch active patient ID set globally by app.py
subject_id = st.session_state.get("subject_id")

if subject_id is None:
    # Fallback to first available patient if not set
    subject_id = valid_admissions_df["subject_id"].iloc[0]

subject_id = int(subject_id)

# Filter valid admissions for THIS specific patient
patient_admissions_df = valid_admissions_df[
    valid_admissions_df["subject_id"] == subject_id
].copy()

if patient_admissions_df.empty:
    st.warning(f"No admission records found for Patient {subject_id}.")
    st.stop()

# Get list of all admission IDs for this patient
available_hadm_ids = sorted(patient_admissions_df["hadm_id"].unique().tolist())

# Resolve active default admission
default_hadm_id = st.session_state.get("hadm_id", available_hadm_ids[0])
default_index = 0
if default_hadm_id in available_hadm_ids:
    default_index = available_hadm_ids.index(default_hadm_id)

# ONLY ONE SELECTBOX: Select Admission ID for the current patient
hadm_id = st.selectbox(
    "Select Admission",
    options=available_hadm_ids,
    index=default_index,
    format_func=lambda value: f"Admission {int(value)}",
)

# Synchronize selections back to session state
st.session_state["subject_id"] = subject_id
st.session_state["hadm_id"] = int(hadm_id)

st.divider()

# =========================================================
# 9. SELECTED PATIENT AND ADMISSION CONTEXT CARDS
# =========================================================

patient_col, admission_col = st.columns(2)

with patient_col:
    st.metric("SELECTED PATIENT", str(subject_id))

with admission_col:
    st.metric("SELECTED ADMISSION", str(hadm_id))
# =========================================================
# 10. RETRIEVE CLINICAL DATA
# =========================================================

try:
    profile_df = get_patient_profile(
        engine,
        subject_id,
        hadm_id,
    )

    diagnoses_df = get_diagnoses(
        engine,
        subject_id,
        hadm_id,
    )

except Exception as exc:
    st.error("Unable to retrieve clinical information.")
    print(f"Clinical data retrieval failed: {exc}")
    st.stop()


if profile_df is None:
    profile_df = pd.DataFrame()
elif not isinstance(profile_df, pd.DataFrame):
    profile_df = pd.DataFrame(profile_df)

if diagnoses_df is None:
    diagnoses_df = pd.DataFrame()
elif not isinstance(diagnoses_df, pd.DataFrame):
    diagnoses_df = pd.DataFrame(diagnoses_df)


# =========================================================
# 11. CLINICAL SUMMARY
# =========================================================

st.subheader("Clinical Summary")

diagnosis_count = len(diagnoses_df)

if not profile_df.empty:

    profile = profile_df.iloc[0]

    age = format_age(
        get_first_available(
            profile,
            ["age", "anchor_age"],
        )
    )

    gender = get_first_available(
        profile,
        ["gender", "sex"],
    )

    admission_type = get_first_available(
        profile,
        ["admission_type", "admittype"],
    )

    admission_location = get_first_available(
        profile,
        ["admission_location", "admit_location"],
    )

    discharge_location = get_first_available(
        profile,
        ["discharge_location", "disch_location"],
    )

    # Patient demographics
    st.markdown("**Patient Demographics**")

    age_col, gender_col = st.columns(2)

    age_col.metric("Age", age)
    gender_col.metric("Gender", gender)

    # Admission summary
    st.markdown("**Admission Summary**")

    type_col, diagnosis_col = st.columns(2)

    type_col.metric("Admission Type", admission_type)

    diagnosis_col.metric(
        "Recorded Diagnoses",
        diagnosis_count,
    )

    # Avoid repeating the demographics and IDs.
    with st.expander("View Detailed Clinical Summary"):

        st.markdown("#### Admission Locations")

        location_col1, location_col2 = st.columns(2)

        with location_col1:
            st.markdown("**Admission Location**")
            st.write(admission_location)

        with location_col2:
            st.markdown("**Discharge Location**")
            st.write(discharge_location)

        st.caption(
            "Details relate to the currently selected admission."
        )

else:
    st.warning(
        "The clinical profile is unavailable for this admission."
    )


# =========================================================
# 12. DIAGNOSIS OVERVIEW
# =========================================================

st.divider()

st.subheader("Diagnosis Overview")

st.caption(
    "Key diagnoses recorded for the currently selected admission."
)

if not diagnoses_df.empty:

    preview_df = diagnoses_df.head(3)

    for _, diagnosis in preview_df.iterrows():

        code = get_diagnosis_code(diagnosis)
        description = get_diagnosis_description(diagnosis)

        with st.container(border=True):

            if description != "N/A":
                st.markdown(f"**ICD {code}**")
                st.write(description)

            else:
                st.markdown(f"**ICD {code}**")
                st.caption(
                    "Description is unavailable in the retrieved data."
                )

    st.caption(
        "Showing up to 3 diagnoses. Visit the dedicated Diagnosis "
        "page for complete records and detailed analysis."
    )

else:
    st.info(
        "No diagnoses were recorded for this admission."
    )


# =========================================================
# 13. ADMISSION HISTORY
# =========================================================

st.divider()

st.subheader("Admission History")

st.caption(
    f"Review recorded admission activity for patient {subject_id}."
)

hide_current_admission = st.checkbox(
    "Exclude the currently selected admission",
    value=True,
    key="hide_current_admission",
)

history_df = patient_admissions_df.copy()

if hide_current_admission:
    history_df = history_df[
        history_df["hadm_id"] != hadm_id
    ]


with st.expander(
    f"View Admission History · {len(history_df)} records",
    expanded=False,
):

    if history_df.empty:

        st.info(
            "No additional admissions are available for this patient."
        )

    else:

        history_display_df = history_df[
            [
                "hadm_id",
                "total_labs",
                "total_meds",
            ]
        ].rename(
            columns={
                "hadm_id": "Admission ID",
                "total_labs": "Distinct Lab Items",
                "total_meds": "Distinct Medications",
            }
        )

        st.dataframe(
            history_display_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "PharmaPal-AI | Clinical data exploration for research purposes. "
    "Clinical information should be interpreted within its "
    "appropriate research and clinical context."
)
