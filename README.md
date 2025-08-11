# Llamalith

FastAPI-powered chat/queue UI with a simple job system, conversation threads, and **secure API access**. The web UI uses session login; the API accepts either a logged-in session **or** a Bearer token (for n8n/curl).

---

## Features

* Web UI under `/chat` for conversations and jobs (Jinja2 templates)
* Queue-based prompt submission to your model worker(s)
* SQLite-backed conversations, messages, and jobs (see `memory.py`)
* **Auth**:

  * Browser/UI: session login (password-protected)
  * API: **session OR Bearer token** (Authorization header)
* Drop-in endpoints for n8n integrations

---

## Quick Start

```bash
# 1) Python 3.10+ recommended
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Set up .env (see below)

# 3) Run
uvicorn main:app --host 0.0.0.0 --port 8000
# UI:    http://localhost:8000/chat
# Docs:  http://localhost:8000/chat/docs (if enabled by your setup)
```

### `.env` example

```ini
# Session and admin login
SECRET_KEY=change_me_please
# Generate a bcrypt hash for the admin password (see snippet below)
ADMIN_PASSWORD_HASH=$2b$12$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# API token for n8n/curl callers
N8N_API_TOKEN=supersecrettokenvalue

# Available model names shown in UI dropdown
LLM_MODELS=mistral,mythomax
```

#### Generate `ADMIN_PASSWORD_HASH`

```bash
python - <<'PY'
import bcrypt, getpass
pwd = getpass.getpass('New admin password: ')
print(bcrypt.hashpw(pwd.encode(), bcrypt.gensalt()).decode())
PY
```

Copy the printed hash into `.env` as `ADMIN_PASSWORD_HASH`.

> **Fail-closed:** The app will raise on boot if `N8N_API_TOKEN` is missing.

---

## Authentication Model

* **UI (browser)** → Session login at `/chat/login` (uses `ADMIN_PASSWORD_HASH`).
* **API** → A route is protected if it uses `require_api_auth`. That guard allows:

  1. a **logged-in session**, or
  2. a valid **Bearer token**: `Authorization: Bearer <N8N_API_TOKEN>` (or `X-API-Token`).

This lets the web UI call `/api/*` without adding headers, while automation clients must send the token.

---

## Endpoints

### Public (session-guarded pages)

* `GET /chat/` → Jobs dashboard (requires session)
* `GET /chat/conversations` → List conversations (session)
* `GET /chat/conversations/{conversation_id}` → Conversation detail (session)
* `GET /chat/jobs` → Jobs page (session)
* `GET /chat/jobs/rows` → Table rows for jobs (session)
* `GET /chat/login` → Login page
* `POST /chat/login` → Submit password, start session
* `GET /chat/logout` → Clear session

> The app is created with `FastAPI(root_path="/chat")`, so URLs in logs may appear without the prefix when proxied. The UI paths above include `/chat`.

### Protected API (session **or** token)

* `GET  /api/conversations/{conversation_id}/messages` → Full message thread
* `GET  /api/conversations/{conversation_id}/latest` → Last user msg, last assistant msg, last job
* `GET  /api/status/{conversation_id}` → **Back-compat:** last assistant text only
* `GET  /api/jobs` → List jobs (optional `status`, `conversation_id`, `limit`)
* `GET  /api/jobs/{job_id}` → Single job details
* `POST /api/conversations/{conversation_id}/reply` → Enqueue a reply in an existing conversation
* `POST /api/jobs` → Create-or-continue a conversation; optionally enqueue work
* `POST /api/submit` → **Legacy alias** equivalent to posting a user message for a given session\_id

All mutations share a single helper internally to add messages and enqueue work, so behavior is consistent.

---

## cURL Examples

Export your token once:

```bash
export TOK="$(grep ^N8N_API_TOKEN .env | cut -d= -f2-)"
```

### Create/continue a conversation and enqueue

```bash
curl -sS -X POST http://localhost:8000/api/jobs \
  -H "Authorization: Bearer $TOK" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "mistral",
        "content": "Write a 2-sentence bedtime story about a helpful dragon.",
        "conversation_id": "abc123",              
        "system_prompt": ""
      }'
```

Response:

```json
{
  "ok": true,
  "queued": true,
  "job_id": 42,
  "conversation_id": "abc123",
  "created_new": false,
  "model": "mistral"
}
```

### Reply to an existing conversation

```bash
curl -sS -X POST http://localhost:8000/api/conversations/abc123/reply \
  -H "Authorization: Bearer $TOK" \
  -H "Content-Type: application/json" \
  -d '{"model":"mistral","content":"Make it rhyme.","system_prompt":""}'
```

### Poll for latest status

```bash
curl -sS http://localhost:8000/api/conversations/abc123/latest \
  -H "Authorization: Bearer $TOK"
```

### Fetch job by id

```bash
curl -sS http://localhost:8000/api/jobs/42 \
  -H "Authorization: Bearer $TOK"
```

---

## n8n Setup (HTTP Request node)

* **Method/URL:** `POST http://<host>:8000/api/jobs`
* **Headers:**

  * `Authorization: Bearer {{$env.N8N_API_TOKEN}}`
  * `Content-Type: application/json`
* **Body (JSON):**

```json
{
  "model": "mistral",
  "content": "Summarize: {{$json.text}}",
  "conversation_id": "{{$json.session_id}}",
  "system_prompt": ""
}
```

* **Polling:** Use another HTTP Request node to `GET /api/conversations/{{$json.conversation_id}}/latest` and check `last_job.status == "done"` before reading `last_assistant.content`.

> Tip: Store the `conversation_id` and `job_id` in your DB (or n8n execution data) for correlation.

---

## Running Behind Nginx (optional)

Minimal reverse proxy example:

```nginx
location /chat/ {
    proxy_pass http://127.0.0.1:8000/chat/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

Ensure the FastAPI app is created with the matching `root_path="/chat"` (already set in `main.py`).

---

## Troubleshooting

* **401 Unauthorized in the web UI**: Make sure routes use `require_api_auth` (not strict token). Ensure your session is active and `SECRET_KEY` is set.
* **401 from curl/n8n**: Missing or wrong token. Send `Authorization: Bearer <N8N_API_TOKEN>`.
* **RuntimeError: N8N\_API\_TOKEN is not set**: Add `N8N_API_TOKEN` to `.env` and restart.
* **Login always fails**: `ADMIN_PASSWORD_HASH` wrong or unset. Regenerate the bcrypt hash.
* **UI paths look odd**: Verify Nginx `location /chat/` and FastAPI `root_path="/chat"` align.
* **Undefined conversation in logs**: Your frontend must pass a real `conversation_id` (the UI templates already include it in links/buttons). The new auth logic also prevents early 401s that previously caused `undefined`.

---

## Project Structure (high level)

```
.
├── main.py                 # Routes (UI + API)
├── auth_utils.py           # Session + token guards
├── memory.py               # DB access: conversations, messages, jobs, queue
├── templates/              # Jinja2 templates
├── static/                 # (optional) static assets
├── requirements.txt
└── .env
```

---

## Changelog

* **2025-08-11**: Added `require_api_auth` (session **or** token) and consolidated enqueue logic.

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

MIT (or your preferred license).




