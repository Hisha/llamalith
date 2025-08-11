from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, Form, Request, HTTPException, Body, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from auth_utils import verify_password, require_login, require_api_auth

from memory import (
    add_message,
    queue_prompt,
    get_db_connection,
    get_conversation_messages,
    list_conversations,
    ensure_conversation,
    create_conversation,
    list_jobs,
    get_job,
    last_model_for_conversation,
    last_system_for_conversation,
)

app = FastAPI(root_path="/chat")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# --------------------------------------------------------------------
# Templates & Globals
# --------------------------------------------------------------------
templates = Jinja2Templates(directory="templates")
templates.env.globals["root_path"] = "/chat/"

AVAILABLE_MODELS = [
    m.strip() for m in os.getenv("LLM_MODELS", "mistral,mythomax").split(",") if m.strip()
]
templates.env.globals["AVAILABLE_MODELS"] = AVAILABLE_MODELS

SYSTEM_PRESETS = [
    {
        "name": "Photo Image Prompt",
        "text": "You are an expert Flux Image Prompt Engineer. Image prompts that stay within the 256 token limit of the Flux box. Photo, hyper-realism, ultra high quality, 8K.",
    },
    {
        "name": "Fantasy Story Author",
        "text": "You are a fantasy author who writes vivid, character-driven short stories with strong worldbuilding and clear arcs.",
    },
    {
        "name": "Non-Fiction Author",
        "text": "You write concise, structured non-fiction with clear headings, examples, and sources when appropriate.",
    },
    {
        "name": "NSFW Fiction Author",
        "text": "You are an experienced NSFW fiction author for consenting adults. Write immersive, character-driven erotically charged scenes focusing on sensuality, emotion, and pacing. Style: natural dialogue, sensory detail (touch/taste/smell/sound/sight), build tension before intimacy, avoid clinical language unless the tone demands it. Maintain continuity and aftercare when appropriate.",
    },
    {
        "name": "Project Planner",
        "text": "Produce a short phased plan with milestones, checklists, risks/mitigations, and success criteria. Keep it implementation-oriented and time-boxed.",
    },
]
templates.env.globals["SYSTEM_PRESETS"] = SYSTEM_PRESETS

# --------------------------------------------------------------------
# Models
# --------------------------------------------------------------------
class CreateJobRequest(BaseModel):
    model: str
    content: str
    conversation_id: str
    system_prompt: str = ""

class ReplyRequest(BaseModel):
    model: str
    content: str
    system_prompt: str = ""

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
def status_badge(status: str) -> str:
    colors = {
        "done": "bg-green-600",
        "processing": "bg-blue-600",
        "queued": "bg-yellow-500",
        "error": "bg-red-600",
        "failed": "bg-red-600",
    }
    dot = colors.get(status.lower(), "bg-gray-500")
    return f"""
      <span class="inline-flex items-center gap-2 px-2 py-1 rounded-full bg-gray-800 border border-gray-700 text-xs">
        <span class="inline-block w-2.5 h-2.5 rounded-full {dot}"></span>
        <span class="uppercase tracking-wide">{status}</span>
      </span>
    """

def enqueue_user_message(conversation_id: str, content: str, model: str, system_prompt: str = "") -> int:
    """Single path to add messages and enqueue a job."""
    if system_prompt:
        add_message(conversation_id, "system", system_prompt)
    add_message(conversation_id, "user", content)
    return queue_prompt(conversation_id, content, model, system_prompt)

# ====================================================================
# GET
# ====================================================================

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("jobs.html", {"request": request, "now": datetime.now})

@app.get("/conversations", response_class=HTMLResponse)
async def conversations_page(request: Request):
    require_login(request)
    convos = list_conversations()
    return templates.TemplateResponse(
        "conversations.html",
        {"request": request, "conversations": convos, "now": datetime.now},
    )

@app.get("/conversations/{conversation_id}", response_class=HTMLResponse)
async def conversation_detail(request: Request, conversation_id: str):
    require_login(request)
    ensure_conversation(conversation_id)
    messages = get_conversation_messages(conversation_id)
    jobs = list_jobs(conversation_id=conversation_id, limit=50)
    last_used_model = last_model_for_conversation(conversation_id)
    last_system_prompt = last_system_for_conversation(conversation_id)
    return templates.TemplateResponse(
        "conversation_detail.html",
        {
            "request": request,
            "conversation_id": conversation_id,
            "messages": messages,
            "jobs": jobs,
            "last_used_model": last_used_model,
            "last_system_prompt": last_system_prompt,
            "now": datetime.now,
        },
    )

@app.get("/jobs", response_class=HTMLResponse)
async def jobs_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("jobs.html", {"request": request, "now": datetime.now})

@app.get("/jobs/rows", response_class=HTMLResponse)
async def jobs_rows(request: Request, status: Optional[str] = None, limit: int = 100):
    # Protect this HTML fragment behind session login
    require_login(request)
    rows = list_jobs(status=status, limit=limit)
    html = []
    for j in rows:
        badge = status_badge(j["status"])
        html.append(
            f"""
          <tr>
            <td>{j['id']}</td>
            <td>{j['conversation_id']}</td>
            <td>{j['model']}</td>
            <td>{badge}</td>
            <td>{j.get('created_at','')}</td>
            <td><button class="underline text-sm" onclick="viewJob({j['id']})">View</button></td>
          </tr>
        """
        )
    return "\n".join(html)

@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "now": datetime.now})

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/chat", status_code=303)

# -------------------- Protected API (token required) --------------------

# Full conversation thread
@app.get("/api/conversations/{conversation_id}/messages", dependencies=[Depends(require_api_auth)])
async def get_conversation(conversation_id: str):
    return {"messages": get_conversation_messages(conversation_id)}

# Latest assistant/user message + last job (handy for polling)
@app.get("/api/conversations/{conversation_id}/latest", dependencies=[Depends(require_api_auth)])
async def get_latest(conversation_id: str):
    msgs = get_conversation_messages(conversation_id) or []
    last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
    last_assistant = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
    jobs = list_jobs(conversation_id=conversation_id, limit=1) or []
    last_job = jobs[0] if jobs else None
    return {"last_user": last_user, "last_assistant": last_assistant, "last_job": last_job}

# Back-compat: previous simple status shape
@app.get("/api/status/{conversation_id}", dependencies=[Depends(require_api_auth)])
async def check_status(conversation_id: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """
        SELECT content
        FROM messages
        WHERE conversation_id = ? AND role = 'assistant'
        ORDER BY timestamp DESC
        LIMIT 1
    """,
        (conversation_id,),
    )
    row = c.fetchone()
    conn.close()
    return {"response": row[0] if row else None}

# List jobs (optionally by status/conversation)
@app.get("/api/jobs", dependencies=[Depends(require_api_auth)])
async def list_jobs_api(
    status: Optional[str] = None,
    conversation_id: Optional[str] = None,
    limit: int = 100,
):
    jobs = list_jobs(conversation_id=conversation_id, status=status, limit=limit)
    return {"jobs": jobs}

# Single job details
@app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_auth)])
async def get_job_api(job_id: int):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# ====================================================================
# POST (Protected API)
# ====================================================================

# Reply within an existing conversation
@app.post("/api/conversations/{conversation_id}/reply", dependencies=[Depends(require_api_auth)])
async def reply_conversation(conversation_id: str, body: ReplyRequest):
    job_id = enqueue_user_message(conversation_id, body.content, body.model, body.system_prompt)
    # Preserve original minimal response shape
    return {"job_id": job_id}

# Create-or-continue conversation and (optionally) enqueue work
@app.post("/api/jobs", dependencies=[Depends(require_api_auth)])
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

    job_id = None
    if content:
        job_id = enqueue_user_message(conversation_id, content, model, system_prompt)

    return {
        "ok": True,
        "queued": bool(job_id),
        "job_id": job_id,
        "conversation_id": conversation_id,
        "created_new": created_new,
        "model": model,
    }

# ====================================================================
# POST (UI - Session login)
# ====================================================================

@app.post("/conversations/{conversation_id}/reply")
async def conversation_reply(
    request: Request,
    conversation_id: str,
    user_input: str = Form(...),
    model: str = Form("mistral"),
    system_prompt: str = Form(""),
):
    require_login(request)
    job_id = enqueue_user_message(conversation_id, user_input, model, system_prompt)
    return RedirectResponse(
        url=f"{request.scope.get('root_path','')}/conversations/{conversation_id}", status_code=303
    )

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, password: str = Form(...)):
    if verify_password(password):
        request.session["logged_in"] = True
        return RedirectResponse(url="/chat", status_code=303)

    return templates.TemplateResponse(
        "login.html", {"request": request, "error": "Invalid password", "now": datetime.now}
    )
