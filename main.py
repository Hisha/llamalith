from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from fastapi import FastAPI, Form, Request, Query, HTTPException, Body
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import uvicorn
from auth_utils import verify_password, require_login
from memory import add_message, queue_prompt, get_db_connection, get_conversation_messages, list_jobs, get_job, list_conversations, ensure_conversation, create_conversation

app = FastAPI(root_path="/chat")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# Templates
templates = Jinja2Templates(directory="templates")
templates.env.globals["root_path"] = "/chat/"

# Request body format for API
class ChatRequest(BaseModel):
    user_input: str
    model: str = "mistral"
    session_id: str
    system_prompt: str = ""

# Create a new job (starts a conversation or continues one)
class CreateJobRequest(BaseModel):
    model: str
    content: str
    conversation_id: str  # you’re already using session_id as conversation_id
    system_prompt: str = ""

# Reply in a conversation -> creates another job
class ReplyRequest(BaseModel):
    model: str
    content: str
    system_prompt: str = ""

#####################################################################################
#                                   GET                                             #
#####################################################################################

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "now": datetime.now
    })

# Get the full conversation thread for a given conversation_id
@app.get("/api/conversations/{conversation_id}/messages")
async def get_conversation(conversation_id: str):
    return {"messages": get_conversation_messages(conversation_id)}

# List recent jobs (optionally by status)
@app.get("/api/jobs")
async def list_jobs_api(status: Optional[str] = None,
                        conversation_id: Optional[str] = None,
                        limit: int = 100):
    # conversation_id comes in as str from URL; keep None or cast if not None
    conv_id = int(conversation_id) if conversation_id is not None and conversation_id.isdigit() else conversation_id
    jobs = list_jobs(conversation_id=conv_id, status=status, limit=limit)
    return {"jobs": jobs}

# Get job details (status + result)
@app.get("/api/jobs/{job_id}")
async def get_job_api(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/api/status/{conversation_id}")
async def check_status(conversation_id: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT content
        FROM messages
        WHERE conversation_id = ? AND role = 'assistant'
        ORDER BY timestamp DESC
        LIMIT 1
    """, (conversation_id,))
    row = c.fetchone()
    conn.close()
    return {"response": row[0] if row else None}

@app.get("/conversations", response_class=HTMLResponse)
async def conversations_page(request: Request):
    require_login(request)
    convos = list_conversations()
    return templates.TemplateResponse("conversations.html", {
        "request": request,
        "conversations": convos,
        "now": datetime.now
    })

@app.get("/conversations/{conversation_id}", response_class=HTMLResponse)
async def conversation_detail(request: Request, conversation_id: str):
    require_login(request)
    ensure_conversation(conversation_id)
    messages = get_conversation_messages(conversation_id)
    jobs = list_jobs(conversation_id=conversation_id, limit=50)
    return templates.TemplateResponse("conversation_detail.html", {
        "request": request,
        "conversation_id": conversation_id,
        "messages": messages,
        "jobs": jobs,
        "now": datetime.now
    })

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "now": datetime.now
    })

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

@app.post("/api/conversations/{conversation_id}/reply")
async def reply_conversation(conversation_id: str, body: ReplyRequest):
    add_message(conversation_id, "user", body.content, model=body.model)
    job_id = queue_prompt(conversation_id, body.content, body.model, body.system_prompt)
    return {"job_id": job_id}

@app.post("/api/jobs")
async def create_job(payload: dict = Body(...)):
    # expected: { "content": str, "model": str, "conversation_id": optional, "system_prompt": optional }
    content = (payload.get("content") or "").strip()
    model = (payload.get("model") or "mistral").strip()
    system_prompt = (payload.get("system_prompt") or "").strip()
    conv_id_raw = (payload.get("conversation_id") or "").strip()

    if not content:
        return JSONResponse({"error": "content is required"}, status_code=400)

    if conv_id_raw:
        # Use provided UUID, ensure conversation exists
        conversation_id = ensure_conversation(conv_id_raw, title=content[:60])
        created_new = False
    else:
        # Create a new UUID conversation
        conversation_id = create_conversation(title=content[:60])
        created_new = True

    add_message(conversation_id, "user", content)
    job_id = queue_prompt(conversation_id, content, model, system_prompt)

    return {
        "ok": True,
        "job_id": job_id,
        "conversation_id": conversation_id,
        "created_new": created_new,
        "model": model,
    }

@app.post("/api/submit")
async def submit_chat(data: ChatRequest):
    conversation_id = data.session_id  # session_id == conversation_id

    # Prevent duplicate consecutive user messages
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT role, content FROM messages
        WHERE conversation_id = ?
        ORDER BY timestamp DESC LIMIT 1
    """, (conversation_id,))
    last_msg = c.fetchone()
    conn.close()

    if not last_msg or last_msg[0] != "user" or last_msg[1] != data.user_input:
        # 👇 persist which model this prompt targets
        add_message(conversation_id, "user", data.user_input, model=data.model)

    job_id = queue_prompt(conversation_id, data.user_input, data.model, data.system_prompt)
    return JSONResponse({"job_id": job_id})

@app.post("/conversations/{conversation_id}/reply")
async def conversation_reply(
    request: Request,
    conversation_id: str,
    user_input: str = Form(...),
    model: str = Form("mistral"),
    system_prompt: str = Form(""),
):
    require_login(request)
    # de-dupe last message logic optional; keep simple:
    add_message(conversation_id, "user", user_input)
    job_id = queue_prompt(conversation_id, user_input, model, system_prompt)
    # back to the conversation page
    return RedirectResponse(
        url=f"{request.scope.get('root_path','')}/conversations/{conversation_id}",
        status_code=303
    )

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
