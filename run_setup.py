from sqlalchemy import text
from db import get_engine

def run_init_script():
    engine = get_engine()
    print("Reading init.sql...")

    with open("init.sql", "r") as f:
        sql_script = f.read()

    print("Executing database initialization script on local PostgreSQL...")
    with engine.begin() as conn:
        conn.execute(text(sql_script))

    print("✅ Local database successfully initialized!")

if __name__ == "__main__":
    run_init_script()