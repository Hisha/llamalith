import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
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
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
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
    return sqlite3.connect(DB_PATH)

# --- Conversation Operations ---
def create_conversation(title):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    conversation_id = c.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def list_conversations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC")
    conversations = c.fetchall()
    conn.close()
    return conversations

# --- Message Memory Operations ---
def add_message(conversation_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
              (conversation_id, role, content))
    conn.commit()
    conn.close()

def get_conversation_messages(conversation_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY timestamp",
              (conversation_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

# --- Queue Operations ---
def queue_prompt(conversation_id, user_input, model, system_prompt):
    conn = sqlite3.connect(DB_PATH)
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
    conn = sqlite3.connect(DB_PATH)
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

def save_assistant_message(conversation_id, content):
    add_message(conversation_id, "assistant", content)

def mark_job_done(job_id, failed=False, result_text=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    status = 'done' if not failed else 'error'
    c.execute("""
        UPDATE chat_queue
        SET status = ?, result = ?, processed_at = ?
        WHERE id = ?
    """, (status, result_text, datetime.utcnow().isoformat(), job_id))
    conn.commit()
    conn.close()
