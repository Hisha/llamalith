import time
import sqlite3
from app.model_runner import run_model
from app.memory import get_db_connection

def process_next_job():
    conn = get_db_connection()
    c = conn.cursor()

    # Get the oldest queued job
    c.execute("""
        SELECT id, conversation_id, user_input, model, system_prompt
        FROM chat_queue
        WHERE status = 'queued'
        ORDER BY created_at ASC
        LIMIT 1
    """)
    job = c.fetchone()

    if not job:
        conn.close()
        return False

    job_id, convo_id, user_input, model, system_prompt = job

    # Update status to "processing"
    c.execute("UPDATE chat_queue SET status = 'processing' WHERE id = ?", (job_id,))
    conn.commit()

    try:
        # Fetch message history
        c.execute("""
            SELECT role, content FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """, (convo_id,))
        messages = c.fetchall()

        # Format messages
        history = [{"role": role, "content": content} for role, content in messages]
        history.append({"role": "user", "content": user_input})

        # Run model
        response = run_model(model, history)

        # Save assistant response
        c.execute("""
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, 'assistant', ?)
        """, (convo_id, response))

        # Mark job complete
        c.execute("UPDATE chat_queue SET status = 'done' WHERE id = ?", (job_id,))
        conn.commit()

        print(f"[✓] Job {job_id} completed.")
    except Exception as e:
        c.execute("UPDATE chat_queue SET status = 'failed' WHERE id = ?", (job_id,))
        conn.commit()
        print(f"[X] Job {job_id} failed: {e}")
    finally:
        conn.close()

    return True


def main_loop(poll_interval=2):
    print("🌀 Chat queue worker started...")
    while True:
        job_found = process_next_job()
        if not job_found:
            time.sleep(poll_interval)

if __name__ == "__main__":
    main_loop()
