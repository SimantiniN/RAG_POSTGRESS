"""
PharmaPal-AI Enhanced RAG Service Module.
Integrates structured patient records, deterministic lab screening and trends, 
clinical evidence search, and Ollama LLM generation.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any
import pandas as pd
import requests
from dotenv import load_dotenv

from services.clinical_queries import (
    get_diagnoses,
    get_labs_for_screening,
    get_medication_summary,
    get_patient_profile,
    search_clinical_evidence,
)
from services.lab_rules import screen_lab_trends, screen_labs

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))


def _df_to_list(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a pandas DataFrame to a clean list of dictionaries with safe date/time serialization."""
    if dataframe is None or dataframe.empty:
        return []
    
    df = dataframe.copy()
    
    for col in df.select_dtypes(include=["datetime", "datetime64", "datetimetz"]).columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
        
    return df.where(df.notna(), None).to_dict(orient="records")


def retrieve_comprehensive_context(engine, subject_id: int, hadm_id: int) -> dict[str, Any]:
    """
    Retrieve structured patient facts, lab summaries, abnormal screening flags, 
    and trend calculations directly from PostgreSQL views and deterministic rules.
    """
    raw_labs_df = get_labs_for_screening(engine, subject_id, hadm_id)
    
    screened_labs_df = screen_labs(raw_labs_df)
    lab_trends_df = screen_lab_trends(raw_labs_df)

    significant_trends = pd.DataFrame()
    if not lab_trends_df.empty:
        significant_trends = lab_trends_df[
            (lab_trends_df["result_count"] > 1) | (lab_trends_df["absolute_change"] != 0)
        ]
        if significant_trends.empty:
            significant_trends = lab_trends_df.head(25)

    abnormal_screenings = pd.DataFrame()
    if not screened_labs_df.empty and "screening_status" in screened_labs_df.columns:
        abnormal_screenings = screened_labs_df[
            screened_labs_df["screening_status"] == "REVIEW"
        ].head(30)

    return {
        "admission_profile": _df_to_list(get_patient_profile(engine, subject_id, hadm_id)),
        "recorded_diagnoses": _df_to_list(get_diagnoses(engine, subject_id, hadm_id)),
        "recent_abnormal_labs": _df_to_list(abnormal_screenings),
        "longitudinal_lab_trends": _df_to_list(significant_trends),
        "prescription_order_summary": _df_to_list(get_medication_summary(engine, subject_id, hadm_id)),
    }


def _format_trends(trends_list: list[dict[str, Any]]) -> str:
    """Format structured lab trends into clear, concise clinical bullet points for the LLM."""
    if not trends_list:
        return "No longitudinal lab trends available for this patient context."

    lines = []
    for t in trends_list:
        name = t.get("lab_name", "Unknown Lab")
        first_val = t.get("first_value")
        latest_val = t.get("latest_value")
        unit = t.get("unit", "")
        change = t.get("absolute_change")
        count = t.get("result_count", 1)

        lines.append(
            f"- {name}: Initial = {first_val} {unit}, Latest = {latest_val} {unit} "
            f"(Net Change: {change}, Total Draws: {count})"
        )
    return "\n".join(lines)


def _format_evidence(evidence_list: list[dict[str, Any]]) -> str:
    """Format retrieved clinical evidence passages into readable context blocks."""
    if not evidence_list:
        return "No reviewed clinical evidence matched this question."

    chunks = []
    for item in evidence_list:
        eid = item.get("evidence_id", "unknown")
        title = item.get("title", "Untitled Guidelines")
        passage = item.get("chunk_text", "")
        chunks.append(f"[E{eid}] Title: {title}\nPassage: {passage}")
        
    return "\n\n".join(chunks)


def generate_rag_answer(
    engine,
    question: str,
    subject_id: int,
    hadm_id: int,
) -> dict[str, Any]:
    """
    Execute the full clinical RAG pipeline:
    1. Fetch structured context & trends from PostgreSQL.
    2. Search unstructured evidence vector/text chunks.
    3. Construct grounded prompts and query Ollama LLM.
    4. Track latency and return full execution payload.
    """
    clean_question = str(question or "").strip()
    if not clean_question:
        raise ValueError("The research question cannot be empty.")

    overall_start = time.perf_counter()

    # 1. Fetch structured database payload & unstructured evidence chunks
    db_start = time.perf_counter()
    patient_context = retrieve_comprehensive_context(engine, subject_id, hadm_id)
    evidence_df = search_clinical_evidence(engine, clean_question, limit=5)
    evidence = _df_to_list(evidence_df)
    db_latency_sec = round(time.perf_counter() - db_start, 3)

    # 2. Prepare Safe Prompt Input Strings
    demo_str = json.dumps(patient_context.get("admission_profile", []), indent=2, default=str)
    diag_str = json.dumps(patient_context.get("recorded_diagnoses", []), indent=2, default=str)
    meds_str = json.dumps(patient_context.get("prescription_order_summary", []), indent=2, default=str)
    labs_str = json.dumps(patient_context.get("recent_abnormal_labs", []), indent=2, default=str)

    # Define Rigorous Clinical System & User Prompts
    system_prompt = """
You are PharmaPal-AI, an expert clinical AI research assistant analyzing MIMIC-IV patient records.

### STRICT CLINICAL GUARDRAILS:
1. DATA GROUNDING: Use the provided patient demographics (Age, Gender, Comorbidities, Admission Type) directly from the database payload to complete the summary vignette. Do NOT output unpopulated placeholders like '[Age]' or '[Gender]' when those values exist in the supplied data.
2. NO ID REDUNDANCY: Do NOT repeat the Subject ID, Admission ID, or placeholder disclaimer notes inside Section 1, as these are already displayed prominently in the UI metrics above.
3. LONGITUDINAL LAB ANALYSIS: Actively analyze multi-draw laboratory trajectory shifts (e.g., shifts in Creatinine, Glucose, Hemoglobin, Hematocrit, PTT, WBC) in Section 2.
4. ABSOLUTE CITATION REQUIREMENT: Cite matching research guidelines using their exact IDs (e.g., [E1], [E2]).
5. REQUIRED DISCLAIMER: Always conclude your output with the mandatory clinical disclaimer.
"""

    user_prompt = f"""
Clinical Question / Goal: {clean_question}

Patient Demographics & Profile Payload:
{demo_str if demo_str != "[]" else "No demographic records found."}

Recorded Diagnoses:
{diag_str if diag_str != "[]" else "No diagnosis codes found."}

Active Medications:
{meds_str if meds_str != "[]" else "No active prescriptions found."}

Recent Abnormal Lab Alerts:
{labs_str if labs_str != "[]" else "No abnormal lab alerts recorded."}

LONGITUDINAL LABORATORY TRAJECTORIES:
{_format_trends(patient_context.get("longitudinal_lab_trends", []))}

Reviewed Clinical Evidence & Guidelines:
{_format_evidence(evidence)}

Provide a rigorous clinical review structured strictly under these headings:

1. Summary of recorded findings
- Start directly with a populated patient vignette using the supplied demographic payload:
  "The patient is a [Age]-year-old [Gender] admitted for [Admission Type] with recorded history of [Key Diagnoses/Comorbidities]."
- Summarize active medications by therapeutic class (e.g., Respiratory, Cardiac, CNS/Sedatives, Gastrointestinal, Ophthalmic, Miscellaneous).
- Do NOT include redundant Subject ID/Admission ID text or placeholder missing-data notes in this section.

2. Relevant laboratory findings & trajectories
- Summarize trajectory trends listed under 'LONGITUDINAL LABORATORY TRAJECTORIES' (initial vs. latest values, net changes, and draw counts).

3. Drug Interactions & Safety Considerations
- Identify potential high-risk drug interactions or contraindications present in this exact regimen.

4. Retrieved research evidence
- Summarize findings from supplied evidence IDs (e.g., [E1], [E2]), or state none matched.

5. Limitations and data gaps
- Note missing clinical parameters or data caveats.

---
Disclaimer: End your output with:
"Disclaimer: PharmaPal-AI is an automated clinical research synthesis tool intended solely for informational and educational support. This output does not constitute medical advice or primary diagnostic evaluation."
"""

    # 4. Query Ollama Local LLM API
    llm_start = time.perf_counter()
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 4096
                },
            },
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. "
            f"Verify that Ollama is running and model '{OLLAMA_MODEL}' is available."
        ) from exc
    llm_latency_sec = round(time.perf_counter() - llm_start, 3)

    payload = response.json()
    answer = payload.get("message", {}).get("content", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty answer string.")

    total_latency_sec = round(time.perf_counter() - overall_start, 3)

    return {
        "answer": answer,
        "patient_context": patient_context,
        "reviewed_evidence": evidence,
        "model": OLLAMA_MODEL,
        "metrics": {
            "db_latency_seconds": db_latency_sec,
            "llm_latency_seconds": llm_latency_sec,
            "total_latency_seconds": total_latency_sec,
        },
    }