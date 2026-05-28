# codeideas_db.py
import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def get_codeideas_connection():
    return pymysql.connect(
        host=os.getenv("CODEIDEAS_DB_HOST"),
        user=os.getenv("CODEIDEAS_DB_USER"),
        password=os.getenv("CODEIDEAS_DB_PASSWORD"),
        database=os.getenv("CODEIDEAS_DB_NAME", "ai_data"),
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

def list_code_ideas(limit=200, status=None, language=None, difficulty=None, search=None):
    conn = get_codeideas_connection()
    try:
        where = []
        params = []

        if status:
            where.append("status = %s")
            params.append(status)

        if language:
            where.append("language = %s")
            params.append(language)

        if difficulty:
            where.append("difficulty = %s")
            params.append(difficulty)

        if search:
            where.append("""
                (
                    module_name LIKE %s OR
                    filename LIKE %s OR
                    purpose LIKE %s OR
                    category LIKE %s
                )
            """)
            term = f"%{search}%"
            params.extend([term, term, term, term])

        where_sql = "WHERE " + " AND ".join(where) if where else ""

        sql = f"""
            SELECT *
            FROM code_ideas
            {where_sql}
            ORDER BY created_at DESC
            LIMIT %s
        """

        params.append(limit)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()

def get_code_idea(idea_id: int):
    conn = get_codeideas_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM code_ideas WHERE id = %s", (idea_id,))
            return cur.fetchone()
    finally:
        conn.close()