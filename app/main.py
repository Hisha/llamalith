from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from app.auth_utils import verify_password, require_login, is_authenticated
from app.memory import add_message_to_conversation, queue_prompt, get_db_connection
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4
import uvicorn

from app.model_runner import run_model
from app.memory import get_session_memory, update_session_memory

app = FastAPI(root_path="/chat")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

DB_PATH = "app/memory.db"

# Jinja2 templates for frontend
templates = Jinja2Templates(directory="app/templates")

# Request body format for API
class ChatRequest(BaseModel):
    user_input: str
    model: str = "mistral"  # "mistral" or "mythomax"
    session_id: str
    system_prompt: str = ""

def insert_user_message(conversation_id: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO messages (conversation_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (conversation_id, "user", content, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def queue_chat_job(conversation_id: str, model: str, system_prompt: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    job_id = str(uuid.uuid4())
    c.execute("INSERT INTO chat_queue (job_id, conversation_id, model, system_prompt, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (job_id, conversation_id, model, system_prompt, "queued", datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()
    return job_id

#####################################################################################
#                                   GET                                             #
#####################################################################################

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("chat_page.html", {
        "request": request,
        "now": datetime.now  # 👈 injects `now` function
    })

@app.get("/api/chat/status/{job_id}")
async def check_job_status(job_id: int):
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT status FROM chat_queue WHERE id = ?", (job_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return JSONResponse(content={"error": "Job not found"}, status_code=404)

    status = row[0]

    if status == "done":
        # Get latest assistant message
        c.execute("""
            SELECT content FROM messages
            WHERE conversation_id = (
                SELECT conversation_id FROM chat_queue WHERE id = ?
            )
            AND role = 'assistant'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (job_id,))
        assistant_row = c.fetchone()
        conn.close()
        return {
            "status": "done",
            "response": assistant_row[0] if assistant_row else "[No response]"
        }

    conn.close()
    return {"status": status}

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request,
        "now": datetime.now
    })

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/chat", status_code=303)

#####################################################################################
#                                   POST                                            #
#####################################################################################

@app.post("/api/submit")
async def submit_chat(data: ChatRequest):
    # 1. Validate or create conversation
    conversation_id = data.session_id  # Using session_id as conversation_id for now

    # 2. Insert user's message into DB
    insert_user_message(conversation_id, data.user_input)

    # 3. Queue the message for processing
    job_id = queue_chat_job(conversation_id, data.model, data.system_prompt)

    return JSONResponse({"job_id": job_id})

@app.get("/api/status/{session_id}")
async def check_status(session_id: str):
    conn = sqlite3.connect("app/memory.db")
    c = conn.cursor()
    c.execute("""
        SELECT content FROM messages
        WHERE session_id = ? AND role = 'assistant'
        ORDER BY timestamp DESC LIMIT 1
    """, (session_id,))
    row = c.fetchone()
    conn.close()

    if row:
        return {"response": row[0]}
    return {"response": None}


@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, password: str = Form(...)):
    if verify_password(password):
        request.session["logged_in"] = True
        return RedirectResponse(url="/chat", status_code=303)

    return templates.TemplateResponse("login.html", {
        "request": request,
        "error": "Invalid password",
        "now": datetime.now
    })

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
