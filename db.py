
import os

import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


@st.cache_resource
def get_engine():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise RuntimeError(
            "DATABASE_URL is not set. Add it to your .env file."
        )

    return create_engine(
        database_url,
        pool_pre_ping=True,
    )


def verify_db_connection():
    """Verify database connectivity and return a status."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))

        return True, "Database connection successful."

    except Exception as exc:
        return False, f"Database connection failed: {exc}"
