# memory.py
import sqlite3
from datetime import datetime

DB_PATH = "app/memory.db"

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

def fetch_next_queued():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, conversation_id, user_input, model, system_prompt
        FROM chat_queue
        WHERE status = 'queued'
        ORDER BY created_at ASC LIMIT 1
    """)
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "id": row[0],
            "conversation_id": row[1],
            "user_input": row[2],
            "model": row[3],
            "system_prompt": row[4]
        }
    return None

def mark_processed(queue_id, result):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        UPDATE chat_queue
        SET status = 'done', result = ?, processed_at = ?
        WHERE id = ?
    """, (result, datetime.utcnow().isoformat(), queue_id))
    conn.commit()
    conn.close()
