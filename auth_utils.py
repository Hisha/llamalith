import os
import bcrypt
import secrets
from typing import Optional
from starlette.middleware.sessions import SessionMiddleware
from fastapi import Request, HTTPException, status, Header

ADMIN_HASH = os.getenv("ADMIN_PASSWORD_HASH")
_API_TOKEN = os.getenv("N8N_API_TOKEN")

def verify_password(password: str) -> bool:
    if not ADMIN_HASH:
        return False
    return bcrypt.checkpw(password.encode(), ADMIN_HASH.encode())

def is_authenticated(request: Request) -> bool:
    return request.session.get("logged_in", False)

def require_login(request: Request):
    if not is_authenticated(request):
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/chat/login"})

# Fail closed so you don't think it's protected when it isn't.
if not _API_TOKEN:
    raise RuntimeError("N8N_API_TOKEN is not set. Add it to .env and restart the server.")

async def require_bearer_token(
    authorization: Optional[str] = Header(default=None),
    x_api_token: Optional[str] = Header(default=None),
) -> None:
    """
    Accepts either:
      - Authorization: Bearer <token>
      - X-API-Token: <token>
    """
    candidate: Optional[str] = None

    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            candidate = parts[1].strip()

    if not candidate and x_api_token:
        candidate = x_api_token.strip()

    if not (candidate and secrets.compare_digest(candidate, _API_TOKEN)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
