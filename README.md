# Llamalith

A lightweight, self-hosted chat + job queue for local LLMs using **FastAPI**, **SQLite**, and **llama.cpp**.
It gives you a small web UI to manage conversations and a simple REST API you can call from tools like **n8n**.

> Default base path is `/chat` (set in `main.py`). So the UI lives at `http://HOST:PORT/chat/` and the API under `http://HOST:PORT/chat/api/...`.

---

## Features

- ✅ Local inference via `llama-cpp-python`
- ✅ Simple conversation store (SQLite) with message history
- ✅ Job queue + background worker(s) for async generation
- ✅ Web UI (Jinja2 templates) with login
- ✅ REST endpoints to submit prompts & poll results (great for n8n)
- ✅ Multi‑model support via `config.json` or env vars

---

## Getting Started

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure models

You can point model **keys** (e.g., `mistral`, `mythomax`) to GGUF files either with a `config.json` **or** environment variables.

**Option A — `config.json`** (recommended):

```jsonc
{
  "model_paths": {
    "mistral":  "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    "mythomax": "models/mythomax-l2-13b.Q4_K_M.gguf"
  }
}
```
Then export:
```bash
export LLAMALITH_CONFIG=config.json
```

**Option B — environment variables:**

```
# Used if LLAMALITH_CONFIG is missing
export MISTRAL_PATH=/abs/path/to/mistral.gguf
export MYTHOMAX_PATH=/abs/path/to/mythomax.gguf
```

Tell the UI which model keys to show in the dropdown:
```bash
export LLM_MODELS="mistral,mythomax"
```

### 3) Set secrets / tuning (optional)

```
# Required for session cookies (UI login)
export SECRET_KEY="some-long-random-string"

# UI admin password (bcrypt hash). Generate one with:
#   python - <<'PY'
import bcrypt;print(bcrypt.hashpw(b"YOUR_PASS", bcrypt.gensalt()).decode())
PY
export ADMIN_PASSWORD_HASH="$2b$12$..."
```

**Generation knobs** (env vars in `model_runner.py`):

```
export LLM_TEMP=0.7
export LLM_TOP_P=0.95
export LLM_TOP_K=40
export LLM_REPEAT_PENALTY=1.1
export LLM_MIROS=0          # mirostat 0/1/2
export LLM_MIROS_TAU=5.0
export LLM_MIROS_ETA=0.1
export LLM_SAFETY_MARGIN=128  # keep some tokens for safety
```

### 4) Run the API & UI

```bash
# dev convenience script
./run.sh
# → serves at http://0.0.0.0:8000/chat/
```

### 5) Run the queue worker

In a separate shell (or as a service):
```bash
./run_worker.sh
```
This starts `queue_worker.py` with multiple subprocesses (`NUM_WORKERS` inside that file).

---

## Data model

- `conversations` — tracks conversation ids/titles
- `messages` — `(conversation_id, role, content, timestamp)`
- `chat_queue` — async work items: `(user_input, model, system_prompt, status, result, created_at, processed_at)`

SQLite DB lives at `memory.db` beside `memory.py` (created automatically).

---

## REST API

> All paths below are **prefixed with `/chat`**.

### Submit a prompt (create job)

`POST /chat/api/submit`

Body:
```json
{
  "user_input": "Write a 3-paragraph bedtime story about a friendly dragon.",
  "model": "mistral",
  "session_id": "story-123",           // this is your conversation_id
  "system_prompt": "You are a gentle storyteller who writes soothing bedtime stories for kids age 4–7."
}
```

Response:
```json
{ "job_id": 42 }
```

### Poll a job

`GET /chat/api/jobs/{job_id}`

Response:
```json
{
  "id": 42,
  "conversation_id": "story-123",
  "user_input": "Write a 3-paragraph bedtime story about a friendly dragon.",
  "model": "mistral",
  "system_prompt": "You are a gentle storyteller...",
  "status": "queued | processing | done | error",
  "result": "…assistant output (present when status=done)…",
  "created_at": "...",
  "processed_at": "..."
}
```

### Quick “latest reply” by conversation (simple polling)

`GET /chat/api/status/{conversation_id}` → returns
```json
{ "response": "last assistant message or null" }
```

### List jobs (optional)

`GET /chat/api/jobs?conversation_id=story-123&status=processing&limit=50`

---

## Example: remote API prompt (curl)

```bash
# 1) submit
JOB_ID=$(curl -sS -H 'Content-Type: application/json'   -d '{
    "user_input": "Write a 2-paragraph bedtime story about a shy comet.",
    "model": "mistral",
    "session_id": "n8n-demo-1",
    "system_prompt": "You are a gentle storyteller; keep vocabulary simple."
  }'   http://localhost:8000/chat/api/submit | jq -r '.job_id')

# 2) poll
while :; do
  STATUS=$(curl -sS http://localhost:8000/chat/api/jobs/$JOB_ID | jq -r '.status')
  if [[ "$STATUS" == "done" || "$STATUS" == "error" ]]; then break; fi
  sleep 2
done

curl -sS http://localhost:8000/chat/api/jobs/$JOB_ID | jq -r '.result'
```

---

## Example: n8n (HTTP nodes)

Minimal workflow idea:

1. **HTTP Request** (POST) → `http://HOST:8000/chat/api/submit`  
   - JSON:
     ```json
     {
       "user_input": "Write a short bedtime story about a curious otter.",
       "model": "mistral",
       "session_id": "story-{{$json.conversationId}}",
       "system_prompt": "You are a gentle storyteller…"
     }
     ```
   - Response → `{{$json.job_id}}`

2. **Wait** (Fixed 2–3s) or **Loop**

3. **HTTP Request** (GET) → `http://HOST:8000/chat/api/jobs/{{$json.job_id}}`  
   - IF `status != done`, loop back to Wait  
   - IF `status == done`, pass `{{$json.result}}` to the next node (e.g., your Kokoro TTS step)

> Tip: if you don’t need strict job tracking, you can skip the job poll and call `GET /chat/api/status/{{conversation_id}}` until it returns non‑null.

---

## Web UI

- `/chat/login` (UI only): requires `ADMIN_PASSWORD_HASH` to be set.
- `/chat/conversations` & `/chat/conversations/{id}`: manage conversations, send prompts from the browser.
- `/chat/jobs`: inspect queued/processing/done jobs.

---

## Project layout

```
.
├── main.py                 # FastAPI app & API routes
├── queue_worker.py         # async job worker (multiprocessing)
├── model_runner.py         # llama.cpp wrapper + model cache
├── memory.py               # SQLite tables & helpers
├── templates/              # Jinja2 templates (UI)
├── run.sh                  # start API/UI
├── run_worker.sh           # start worker(s)
└── requirements.txt
```

---

## Operational notes

- **Base path**: The app uses `root_path="/chat"`. If you reverse-proxy (nginx/Caddy), forward that base path or set another one.
- **Workers**: edit `NUM_WORKERS` in `queue_worker.py` to scale CPU use; set `n_threads` in `model_runner.get_model()` accordingly.
- **SQLite**: for more concurrency, enable WAL mode or move to Postgres (you’d swap the small `memory.py` layer).

---

## Roadmap / Ideas for improvement

- **CORS** config for remote callers (n8n, browser apps).
- **Streaming responses** (Server-Sent Events) to avoid polling.
- **Job cancellation** endpoint and timeouts.
- **Config file schema + validation** (pydantic) for `config.json`.
- **Dockerfile & compose** with volumes for models and DB.
- **Observability**: structured logging (JSON), request IDs, `/healthz`.
- **Prompt templates**: store named system prompts, versioning.
- **Retry on failure** in `queue_worker.py` with exponential backoff.
- **User/API keys**: multi‑user support and per‑key rate limits.

---

## License

See [LICENSE](LICENSE).
