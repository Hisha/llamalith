from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from app.auth_utils import verify_password, require_login, is_authenticated
from app.memory import add_message_to_conversation, queue_prompt
from pydantic import BaseModel
from datetime import datetime
import uvicorn

from app.model_runner import run_model
from app.memory import get_session_memory, update_session_memory

app = FastAPI(root_path="/chat")
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY"))

# Jinja2 templates for frontend
templates = Jinja2Templates(directory="app/templates")

# Request body format for API
class ChatRequest(BaseModel):
    user_input: str
    model: str = "mistral"  # "mistral" or "mythomax"
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
        "now": datetime.now  # 👈 injects `now` function
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

@app.post("/api/chat")
async def chat_api(data: ChatRequest):
    # Save user message to DB
    add_message_to_conversation(data.session_id, "user", data.user_input)

    # Queue it for background processing
    queue_id = queue_prompt(
        session_id=data.session_id,
        user_input=data.user_input,
        model=data.model,
        system_prompt=data.system_prompt
    )

    return {"status": "queued", "queue_id": queue_id}

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
