"""
Database Query Service for PharmaPal-AI.
Fetches structured patient profiles, diagnoses, lab events, prescriptions,
and pgvector semantic clinical evidence directly from PostgreSQL.
"""

import pandas as pd
import streamlit as st
from sqlalchemy import inspect, text
from sentence_transformers import SentenceTransformer


# --------------------------------------------------------------------------
# Embedding Model Cache for pgvector
# --------------------------------------------------------------------------
@st.cache_resource
def load_embed_model():
    """Loads the sentence transformer model once for vector search."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def get_patient_profile(engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Retrieve patient demographic and admission details."""
    sql = text("""
        SELECT 
            p.subject_id,
            p.gender,
            p.anchor_age AS age,
            a.hadm_id,
            a.race,
            a.admission_type,
            a.admission_location
        FROM admissions a
        LEFT JOIN patients p ON a.subject_id = p.subject_id
        WHERE a.subject_id = :subject_id AND a.hadm_id = :hadm_id;
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"subject_id": subject_id, "hadm_id": hadm_id})
    except Exception as err:
        print(f"[Warning] Error fetching patient profile: {err}")
        return pd.DataFrame()


# def get_patient_admissions(engine, subject_id: int = None) -> pd.DataFrame:
#     """Retrieve patient admissions along with mortality status."""
#     if subject_id:
#         sql = text("""
#             SELECT 
#                 a.subject_id,
#                 a.hadm_id, 
#                 a.admission_type, 
#                 a.admission_location, 
#                 a.race,
#                 a.admittime,
#                 a.dischtime,
#                 a.deathtime,
#                 a.hospital_expire_flag,
#                 p.dod,
#                 p.gender
#             FROM admissions a
#             JOIN patients p ON a.subject_id = p.subject_id
#             WHERE a.subject_id = :subject_id
#             ORDER BY a.hadm_id DESC;
#         """)
#         params = {"subject_id": subject_id}
#     else:
#         # Cohort overview fallback query
#         sql = text("""
#             SELECT 
#                 a.subject_id, 
#                 a.hadm_id, 
#                 a.hospital_expire_flag,
#                 p.dod
#             FROM admissions a
#             JOIN patients p ON a.subject_id = p.subject_id
#             ORDER BY a.subject_id ASC
#             LIMIT 100;
#         """)
#         params = {}
def get_patient_admissions(engine, subject_id: int = None) -> pd.DataFrame:
    """Retrieve patient admissions along with mortality status."""
    if subject_id:
        sql = text("""
            SELECT 
                a.subject_id,
                a.hadm_id, 
                a.admission_type, 
                a.admission_location, 
                a.race,
                a.admittime,
                a.dischtime,
                a.deathtime,
                a.hospital_expire_flag,
                p.dod,
                p.gender
            FROM admissions a
            JOIN patients p ON a.subject_id = p.subject_id
            WHERE a.subject_id = :subject_id
            ORDER BY a.hadm_id DESC;
        """)
        params = {"subject_id": subject_id}
    else:
        # Load ALL admissions and patients without limiting to 100
        sql = text("""
            SELECT 
                a.subject_id, 
                a.hadm_id, 
                a.hospital_expire_flag,
                p.dod
            FROM admissions a
            JOIN patients p ON a.subject_id = p.subject_id
            ORDER BY a.subject_id ASC;
        """)
        params = {}
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
    except Exception as err:
        print(f"[Warning] Error fetching patient admissions: {err}")
        return pd.DataFrame()


def get_patient_summary_all_tables(engine, subject_id: int, hadm_id: int) -> dict:
    """
    Dynamically introspect all database tables containing subject_id and hadm_id columns 
    to compile a count of records available for this specific admission.
    """
    summary = {}
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        with engine.connect() as conn:
            for table in table_names:
                columns = [col['name'] for col in inspector.get_columns(table)]
                
                has_subject = 'subject_id' in columns
                has_hadm = 'hadm_id' in columns

                if has_subject and has_hadm:
                    query = text(f"SELECT COUNT(*) FROM {table} WHERE subject_id = :s_id AND hadm_id = :h_id")
                    res = conn.execute(query, {"s_id": subject_id, "h_id": hadm_id}).scalar()
                    if res > 0:
                        summary[table] = res
                elif has_subject:
                    query = text(f"SELECT COUNT(*) FROM {table} WHERE subject_id = :s_id")
                    res = conn.execute(query, {"s_id": subject_id}).scalar()
                    if res > 0:
                        summary[f"{table} (subject-only)"] = res

        return summary
    except Exception as err:
        print(f"[Warning] Error during multi-table introspection: {err}")
        return summary


def get_diagnoses(engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Retrieve recorded ICD diagnoses for the given admission."""
    sql = text("""
        SELECT 
            d.icd_code,
            d.icd_version,
            COALESCE(dict.long_title, 'Diagnosis Code ' || d.icd_code) AS diagnosis_title
        FROM diagnoses_icd d
        LEFT JOIN d_icd_diagnoses dict 
          ON d.icd_code = dict.icd_code AND d.icd_version = dict.icd_version
        WHERE d.subject_id = :subject_id AND d.hadm_id = :hadm_id;
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"subject_id": subject_id, "hadm_id": hadm_id})
    except Exception as err:
        print(f"[Warning] Error fetching diagnoses: {err}")
        return pd.DataFrame()


def get_medication_summary(engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Retrieve active prescription records for the admission."""
    sql = text("""
        SELECT DISTINCT
            p.drug,
            p.dose_val_rx AS dose,
            p.dose_unit_rx AS unit,
            p.route,
            p.starttime,
            p.stoptime
        FROM prescriptions_new_clean p
        WHERE p.subject_id = :subject_id 
          AND p.hadm_id = :hadm_id
          AND p.drug IS NOT NULL
        LIMIT 50;
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"subject_id": subject_id, "hadm_id": hadm_id})
    except Exception as err:
        print(f"[Warning] Error fetching prescriptions: {err}")
        return pd.DataFrame()


def get_medication_orders(engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Retrieve detailed prescription orders for the admission."""
    return get_medication_summary(engine, subject_id, hadm_id)


def get_labs_for_screening(engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Retrieve lab event values, reference ranges, and timestamps using vw_admission_labs."""
    sql = text("""
        SELECT 
            subject_id,
            hadm_id,
            itemid,
            lab_name,
            valuenum,
            unit AS valueuom,
            charttime,
            ref_range_lower,
            ref_range_upper,
            source_flag
        FROM vw_admission_labs
        WHERE subject_id = :subject_id 
          AND hadm_id = :hadm_id 
          AND valuenum IS NOT NULL
        ORDER BY charttime ASC;
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"subject_id": subject_id, "hadm_id": hadm_id})
    except Exception as err:
        print(f"[Warning] Error fetching lab events from view: {err}")
        return pd.DataFrame()


def get_lab_summary(engine, subject_id: int, hadm_id: int) -> pd.DataFrame:
    """Retrieve aggregated lab counts and metrics using vw_admission_labs."""
    sql = text("""
        SELECT 
            itemid,
            lab_name,
            COUNT(valuenum) AS total_draws,
            MIN(valuenum) AS min_val,
            MAX(valuenum) AS max_val,
            MAX(unit) AS unit,
            MAX(ref_range_lower) AS ref_range_lower,
            MAX(ref_range_upper) AS ref_range_upper,
            COUNT(CASE WHEN valuenum < ref_range_lower OR valuenum > ref_range_upper THEN 1 END) AS abnormal_count
        FROM vw_admission_labs
        WHERE subject_id = :subject_id 
          AND hadm_id = :hadm_id 
          AND valuenum IS NOT NULL
        GROUP BY itemid, lab_name
        ORDER BY abnormal_count DESC, total_draws DESC;
    """)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params={"subject_id": subject_id, "hadm_id": hadm_id})
    except Exception as err:
        print(f"[Warning] Error fetching lab summary from view: {err}")
        return pd.DataFrame()


def get_lab_history(engine, subject_id: int, hadm_id: int, itemid: int = None) -> pd.DataFrame:
    """Retrieve longitudinal sequence of lab results for plotting and analysis."""
    if itemid:
        sql = text("""
            SELECT 
                l.charttime,
                COALESCE(d.label, 'Lab Item ' || l.itemid::text) AS lab_name,
                l.valuenum,
                l.valueuom,
                l.flag
            FROM labevents l
            LEFT JOIN d_labitems d ON l.itemid = d.itemid
            WHERE l.subject_id = :subject_id 
              AND l.hadm_id = :hadm_id 
              AND l.itemid = :itemid
              AND l.valuenum IS NOT NULL
            ORDER BY l.charttime ASC;
        """)
        params = {"subject_id": subject_id, "hadm_id": hadm_id, "itemid": itemid}
    else:
        sql = text("""
            SELECT 
                l.charttime,
                COALESCE(d.label, 'Lab Item ' || l.itemid::text) AS lab_name,
                l.valuenum,
                l.valueuom,
                l.flag
            FROM labevents l
            LEFT JOIN d_labitems d ON l.itemid = d.itemid
            WHERE l.subject_id = :subject_id 
              AND l.hadm_id = :hadm_id 
              AND l.valuenum IS NOT NULL
            ORDER BY l.charttime ASC;
        """)
        params = {"subject_id": subject_id, "hadm_id": hadm_id}

    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
    except Exception as err:
        print(f"[Warning] Error fetching lab history: {err}")
        return pd.DataFrame()


def retrieve_vector_context(engine, question: str, subject_id: int, hadm_id: int, top_k: int = 5) -> pd.DataFrame:
    """
    Retrieves semantic clinical passage embeddings directly from PostgreSQL 
    using the pgvector cosine distance operator (<=>) filtered by patient admission.
    """
    model = load_embed_model()
    query_vector = model.encode(question).tolist()
    
    sql = text("""
        SELECT 
            content, 
            source_table, 
            (embedding <=> :query_vector::vector) AS distance
        FROM clinical_embeddings
        WHERE subject_id = :subject_id AND hadm_id = :hadm_id
        ORDER BY distance ASC
        LIMIT :top_k;
    """)
    
    try:
        with engine.connect() as conn:
            return pd.read_sql(
                sql, 
                conn, 
                params={
                    "query_vector": str(query_vector), 
                    "subject_id": int(subject_id), 
                    "hadm_id": int(hadm_id), 
                    "top_k": int(top_k)
                }
            )
    except Exception as err:
        print(f"[Warning] Error executing pgvector retrieval: {err}")
        return pd.DataFrame()


def search_clinical_evidence(engine, query_text: str, limit: int = 5) -> pd.DataFrame:
    """Dynamically search matching guidelines from clinical_evidence_chunks."""
    clean_query = str(query_text or "").strip()
    if not clean_query:
        return pd.DataFrame()

    tokens = [token for token in clean_query.split() if len(token) > 3]
    if not tokens:
        tokens = [clean_query]

    conditions = " OR ".join([f"chunk_text ILIKE :token_{i}" for i in range(len(tokens))])
    
    sql = text(f"""
        SELECT 
            evidence_id, 
            title, 
            chunk_text
        FROM clinical_evidence_chunks
        WHERE {conditions}
        LIMIT :limit;
    """)

    params = {f"token_{i}": f"%{token}%" for i, token in enumerate(tokens)}
    params["limit"] = limit

    try:
        with engine.connect() as conn:
            return pd.read_sql(sql, conn, params=params)
    except Exception as err:
        print(f"[Notice] Clinical evidence search skipped: {err}")
        return pd.DataFrame()