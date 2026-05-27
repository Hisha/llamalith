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

def list_code_ideas(limit=200):
    conn = get_codeideas_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT *
                FROM code_ideas
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
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