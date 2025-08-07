from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from app.model_runner import run_model
from app.memory import get_session_memory, update_session_memory

app = FastAPI()

# Mount static files (JS/CSS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates for frontend
templates = Jinja2Templates(directory="app/templates")

# Request body format for API
class ChatRequest(BaseModel):
    user_input: str
    model: str = "mistral"  # "mistral" or "mythomax"
    session_id: str
    system_prompt: str = ""

@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/api/chat")
async def chat_api(data: ChatRequest):
    # Load memory for session
    history = get_session_memory(data.session_id)

    # Build prompt
    messages = []
    if data.system_prompt:
        messages.append({"role": "system", "content": data.system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": data.user_input})

    # Run model
    output = run_model(data.model, messages)

    # Update memory
    update_session_memory(data.session_id, {"role": "user", "content": data.user_input})
    update_session_memory(data.session_id, {"role": "assistant", "content": output})

    return JSONResponse({"reply": output})

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
