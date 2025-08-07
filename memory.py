import os
import json

MEMORY_DIR = "app/memory_store"
os.makedirs(MEMORY_DIR, exist_ok=True)

def _get_memory_file(session_id):
    return os.path.join(MEMORY_DIR, f"{session_id}.json")

def get_session_memory(session_id):
    path = _get_memory_file(session_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def update_session_memory(session_id, message):
    path = _get_memory_file(session_id)
    history = get_session_memory(session_id)
    history.append(message)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
