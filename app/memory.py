import sqlite3
import os
from datetime import datetime

DB_PATH = "app/memory.db"

# Ensure database and table exist
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Fetch all messages for a session
def get_session_memory(session_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM memory WHERE session_id = ? ORDER BY timestamp", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# Append a message to session memory
def update_session_memory(session_id, message):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO memory (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, message["role"], message["content"], datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
