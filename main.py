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
from memory import add_message, queue_prompt, get_db_connection, get_conversation_messages, list_conversations, ensure_conversation, create_conversation, list_jobs, get_job, last_model_for_conversation, last_system_for_conversation

app = FastAPI(root_path="/chat")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# Templates
templates = Jinja2Templates(directory="templates")
templates.env.globals["root_path"] = "/chat/"

AVAILABLE_MODELS = [m.strip() for m in os.getenv("LLM_MODELS", "mistral,mythomax").split(",") if m.strip()]
templates.env.globals["AVAILABLE_MODELS"] = AVAILABLE_MODELS

SYSTEM_PRESETS = [
    {
        "name": "Photo Image Prompt",
        "text": "You are an expert Flux Image Prompt Engineer. Image prompts that stay within the 256 token limit of the Flux box. Photo, hyper-realism, ultra high quality, 8K."
    },
    {
        "name": "Fantasy Story Author",
        "text": "You are a fantasy author who writes vivid, character-driven short stories with strong worldbuilding and clear arcs."
    },
    {
        "name": "Non-Fiction Author",
        "text": "You write concise, structured non-fiction with clear headings, examples, and sources when appropriate."
    },
    {
        "name": "NSFW Fiction Author",
        "text": "You are an experienced NSFW fiction author for consenting adults. Write immersive, character-driven erotically charged scenes focusing on sensuality, emotion, and pacing. Style: natural dialogue, sensory detail (touch/taste/smell/sound/sight), build tension before intimacy, avoid clinical language unless the tone demands it. Maintain continuity and aftercare when appropriate."
    },
    {
        "name": "Project Planner",
        "text": "Produce a short phased plan with milestones, checklists, risks/mitigations, and success criteria. Keep it implementation-oriented and time-boxed."
    },
]
templates.env.globals["SYSTEM_PRESETS"] = SYSTEM_PRESETS

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

def status_badge(status: str) -> str:
    colors = {
        "done":       "bg-green-600",
        "processing": "bg-blue-600",
        "queued":     "bg-yellow-500",
        "error":      "bg-red-600",
        "failed":     "bg-red-600",
    }
    dot = colors.get(status.lower(), "bg-gray-500")
    # dot + pill text
    return f'''
      <span class="inline-flex items-center gap-2 px-2 py-1 rounded-full bg-gray-800 border border-gray-700 text-xs">
        <span class="inline-block w-2.5 h-2.5 rounded-full {dot}"></span>
        <span class="uppercase tracking-wide">{status}</span>
      </span>
    '''

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
    jobs = list_jobs(conversation_id=conversation_id, status=status, limit=limit)
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
    last_used_model = last_model_for_conversation(conversation_id)
    last_system_prompt = last_system_for_conversation(conversation_id)
    return templates.TemplateResponse("conversation_detail.html", {
        "request": request,
        "conversation_id": conversation_id,
        "messages": messages,
        "jobs": jobs,
        "last_used_model": last_used_model,
        "last_system_prompt": last_system_prompt,
        "now": datetime.now
    })

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("jobs.html", {
        "request": request,
        "now": datetime.now
    })

@app.get("/jobs/rows", response_class=HTMLResponse)
async def jobs_rows(status: Optional[str] = None, limit: int = 100):
    rows = list_jobs(status=status, limit=limit)
    html = []
    for j in rows:
        badge = status_badge(j["status"])
        html.append(f"""
          <tr>
            <td>{j['id']}</td>
            <td>{j['conversation_id']}</td>
            <td>{j['model']}</td>
            <td>{badge}</td>
            <td>{j.get('created_at','')}</td>
            <td><button class="underline text-sm" onclick="viewJob({j['id']})">View</button></td>
          </tr>
        """)
    return "\n".join(html)

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
    add_message(conversation_id, "user", body.content)  # ← remove model=...
    job_id = queue_prompt(conversation_id, body.content, body.model, body.system_prompt)
    return {"job_id": job_id}

@app.post("/api/jobs")
async def create_job(payload: dict = Body(...)):
    content = (payload.get("content") or "").strip()
    model = (payload.get("model") or "mistral").strip()
    system_prompt = (payload.get("system_prompt") or "").strip()
    conv_id_raw = (payload.get("conversation_id") or "").strip()

    if not content and not system_prompt:
        return JSONResponse({"error": "content or system_prompt is required"}, status_code=400)

    title_seed = (content or system_prompt)[:60]
    if conv_id_raw:
        conversation_id = ensure_conversation(conv_id_raw, title=title_seed)
        created_new = False
    else:
        conversation_id = create_conversation(title=title_seed)
        created_new = True

    if system_prompt:
        add_message(conversation_id, "system", system_prompt)

    job_id = None
    if content:
        add_message(conversation_id, "user", content)
        job_id = queue_prompt(conversation_id, content, model, system_prompt)

    return {
        "ok": True,
        "queued": bool(job_id),
        "job_id": job_id,
        "conversation_id": conversation_id,
        "created_new": created_new,
        "model": model,
    }

@app.post("/api/submit")
async def submit_chat(data: ChatRequest):
    conversation_id = data.session_id
    # ... keep your de-dupe query ...
    if not last_msg or last_msg[0] != "user" or last_msg[1] != data.user_input:
        add_message(conversation_id, "user", data.user_input)  # ← remove model=...

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
