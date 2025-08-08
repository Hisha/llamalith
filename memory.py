# memory.py
import os
import sqlite3
from datetime import datetime
from uuid import uuid4

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()

    # Conversations: id is TEXT (UUID)
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Messages: conversation_id TEXT (FK-ish)
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Queue: conversation_id TEXT
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
            processed_at TEXT
        )
    """)

    conn.commit()
    conn.close()

init_db()

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# --- Conversation Operations ---
def ensure_conversation(conversation_id: str, title: str = "") -> str:
    """Create the conversation if it doesn't exist. Return conversation_id."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,))
    exists = c.fetchone()
    if not exists:
        c.execute(
            "INSERT INTO conversations (id, title) VALUES (?, ?)",
            (conversation_id, title or "New Conversation")
        )
        conn.commit()
    conn.close()
    return conversation_id

def create_conversation(title: str = "") -> str:
    """Create a brand new conversation with a generated UUID."""
    cid = str(uuid4())
    ensure_conversation(cid, title=title)
    return cid

def list_conversations():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
    conversations = c.fetchall()
    conn.close()
    return conversations

# --- Message Memory Operations ---
def add_message(conversation_id: str, role: str, content: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
        (conversation_id, role, content)
    )
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
        (conversation_id,)
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# --- Queue Operations ---
def queue_prompt(conversation_id: str, user_input: str, model: str, system_prompt: str):
    conn = get_db_connection()
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
    conn = get_db_connection()
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
        c.execute("UPDATE chat_queue SET status = 'processing' WHERE id = ?", (job_id,))
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

def save_assistant_message(conversation_id: str, content: str):
    add_message(conversation_id, "assistant", content)

def mark_job_done(job_id: int, failed=False, result_text=None):
    conn = get_db_connection()
    c = conn.cursor()
    status = 'done' if not failed else 'error'
    c.execute("""
        UPDATE chat_queue
        SET status = ?, result = ?, processed_at = ?
        WHERE id = ?
    """, (status, result_text, datetime.utcnow().isoformat(), job_id))
    conn.commit()
    conn.close()
