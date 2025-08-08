from dotenv import load_dotenv
load_dotenv()

import os
import sqlite3
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from datetime import datetime
import uvicorn

from auth_utils import verify_password, require_login
from memory import add_message, queue_prompt, get_db_connection

app = FastAPI(root_path="/chat")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# Templates
templates = Jinja2Templates(directory="templates")

# Request body format for API
class ChatRequest(BaseModel):
    user_input: str
    model: str = "mistral"
    session_id: str
    system_prompt: str = ""

#####################################################################################
#                                   GET                                             #
#####################################################################################

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    require_login(request)
    return templates.TemplateResponse("chat_page.html", {
        "request": request,
        "now": datetime.now
    })

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
    conversation_id = data.session_id  # session_id == conversation_id

    # Save user's message
    add_message(conversation_id, "user", data.user_input)

    # Queue the job
    job_id = queue_prompt(conversation_id, data.user_input, data.model, data.system_prompt)

    return JSONResponse({"job_id": job_id})

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
