import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            model TEXT,
            content TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            user_input TEXT,
            model TEXT,
            system_prompt TEXT,
            status TEXT DEFAULT 'queued',
            result TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            processed_at TEXT,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# --- Conversation Operations ---
def create_conversation(title):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    conversation_id = c.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def list_conversations():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
    conversations = c.fetchall()
    conn.close()
    return conversations

# --- Message Memory Operations ---
def add_message(conversation_id, role, content, model=None):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, model, content) VALUES (?, ?, ?, ?)",
              (conversation_id, role, content, model))
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
              (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# --- Queue Operations ---
def queue_prompt(conversation_id, user_input, model, system_prompt):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_queue (conversation_id, user_input, model, system_prompt)
        VALUES (?, ?, ?, ?)
    """, (conversation_id, user_input, model, system_prompt))
    queue_id = c.lastrowid
    conn.commit()
    conn.close()
    return queue_id

def claim_next_job():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.isolation_level = None
    c = conn.cursor()
    try:
        c.execute("BEGIN IMMEDIATE")
        c.execute("""
            SELECT id, conversation_id, user_input, model, system_prompt
            FROM chat_queue
            WHERE status = 'queued'
            ORDER BY created_at ASC
            LIMIT 1
        """)
        row = c.fetchone()
        if not row:
            c.execute("COMMIT")
            conn.close()
            return None

        job_id, convo_id, user_input, model, system_prompt = row
        c.execute("""
            UPDATE chat_queue
            SET status = 'processing'
            WHERE id = ?
        """, (job_id,))
        c.execute("COMMIT")
        conn.close()
        return {
            "id": job_id,
            "conversation_id": convo_id,
            "user_input": user_input,
            "model": model,
            "system_prompt": system_prompt
        }
    except Exception:
        c.execute("ROLLBACK")
        conn.close()
        raise

def save_assistant_message(conversation_id, content, model=None):
    add_message(conversation_id, "assistant", content, model=model)

def mark_job_done(job_id, failed=False, result_text=None):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    status = 'done' if not failed else 'error'
    c.execute("""
        UPDATE chat_queue
        SET status = ?, result = ?, processed_at = ?
        WHERE id = ?
    """, (status, result_text, datetime.utcnow().isoformat(), job_id))
    conn.commit()
    conn.close()

# --- Jobs (chat_queue) helpers ---

def list_jobs(status=None, limit=50):
    conn = get_db_connection()
    c = conn.cursor()
    if status:
        c.execute("""
            SELECT id, conversation_id, model, status, created_at, processed_at
            FROM chat_queue
            WHERE status = ?
            ORDER BY id DESC
            LIMIT ?
        """, (status, limit))
    else:
        c.execute("""
            SELECT id, conversation_id, model, status, created_at, processed_at
            FROM chat_queue
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "conversation_id": r[1],
            "model": r[2],
            "status": r[3],
            "created_at": r[4],
            "processed_at": r[5],
        } for r in rows
    ]

def get_job(job_id: int):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT id, conversation_id, user_input, model, system_prompt, status, result, created_at, processed_at
        FROM chat_queue
        WHERE id = ?
    """, (job_id,))
    r = c.fetchone()
    conn.close()
    if not r:
        return None
    return {
        "id": r[0],
        "conversation_id": r[1],
        "user_input": r[2],
        "model": r[3],
        "system_prompt": r[4],
        "status": r[5],
        "result": r[6],
        "created_at": r[7],
        "processed_at": r[8],
    }

# --- Extra helpers for conversations / jobs ---

def get_conversation(conversation_id: str):
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations WHERE id = ?", (conversation_id,))
    row = c.fetchone()
    conn.close()
    return row  # (id, title, created_at) or None

def list_jobs(conversation_id: Optional[int] = None,
              status: Optional[str] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()

    where = []
    params = []

    if conversation_id is not None:
        where.append("conversation_id = ?")
        params.append(conversation_id)
    if status is not None:
        where.append("status = ?")
        params.append(status)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    c.execute(f"""
        SELECT id, conversation_id, model, status, created_at, processed_at
        FROM chat_queue
        {where_sql}
        ORDER BY created_at DESC
        LIMIT ?
    """, (*params, limit))
    rows = c.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "conversation_id": r[1],
            "model": r[2],
            "status": r[3],
            "created_at": r[4],
            "processed_at": r[5],
        }
        for r in rows
    ]

def ensure_conversation(conversation_id: str):
    """Create a placeholder conversation row if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id FROM conversations WHERE id = ?", (conversation_id,))
    row = c.fetchone()
    if not row:
        c.execute("INSERT INTO conversations (id, title) VALUES (?, ?)", (conversation_id, f"Session {conversation_id}"))
        conn.commit()
    conn.close()

