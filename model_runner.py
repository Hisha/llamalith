import os
import json
import logging 
from typing import List, Dict, Any
from llama_cpp import Llama

SAFETY_MARGIN = int(os.getenv("LLM_SAFETY_MARGIN", "128"))

# ---------- config ----------
CONFIG_PATH = os.getenv("LLAMALITH_CONFIG", "config.json")
MODEL_PATHS: Dict[str, str] = {}
MODEL_FORMATS: Dict[str, str] = {}
MODEL_SETTINGS: Dict[str, Dict[str, Any]] = {}

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)
    MODEL_PATHS    = cfg.get("model_paths", {}) or {}
    MODEL_FORMATS  = cfg.get("model_formats", {}) or {}
    MODEL_SETTINGS = cfg.get("model_settings", {}) or {}
else:
    MODEL_PATHS = {
        "mistral":  os.getenv("MISTRAL_PATH",  "models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "mythomax": os.getenv("MYTHOMAX_PATH", "models/mythomax/mythomax-l2-13b.Q4_K_M.gguf"),
        "openchat": os.getenv("OPENCHAT_PATH", "models/openchat/openchat-3.5-1210.Q8_0.gguf"),
    }
    MODEL_FORMATS = {
        "mistral":  "mistral-instruct",
        "mythomax": "chatml",
        "openchat": "chatml",
    }

# ---------- model cache ----------
_LOADED: Dict[str, Llama] = {}

def _settings_for(model_key: str) -> Dict[str, Any]:
    return MODEL_SETTINGS.get(model_key, {}) or {}

def get_model(model_key: str) -> Llama:
    """Singleton loader for each model key."""
    if model_key in _LOADED:
        return _LOADED[model_key]

    path = MODEL_PATHS.get(model_key)
    if not path or not os.path.exists(path):
        raise ValueError(f"Model '{model_key}' not found or path does not exist: {path!r}")

    s = _settings_for(model_key)
    # allow per-model n_ctx; default 4096
    n_ctx = int(os.getenv("LLM_N_CTX", str(s.get("n_ctx", 4096))))

    # chat_format: env overrides config, which overrides fallback
    model_format = os.getenv("LLM_CHAT_FORMAT") or MODEL_FORMATS.get(model_key) or "llama-2"

    llm = Llama(
        model_path=path,
        chat_format=model_format,
        n_ctx=n_ctx,
        n_threads=os.cpu_count() or 8,
        n_batch=int(os.getenv("LLM_N_BATCH", "512")),
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    _LOADED[model_key] = llm
    print(f"[llamalith] loaded model={model_key} path={path} model_format={model_format} n_ctx={n_ctx}")
    return llm

# ---------- prompt helpers ----------
def format_messages(messages: List[Dict[str, str]]) -> str:
    """Plain string representation (only for token counting)."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            parts.append(f"[USER]\n{content}\n")
        elif role == "assistant":
            parts.append(f"[ASSISTANT]\n{content}\n")
        else:
            parts.append(f"[{role.upper()}]\n{content}\n")
    parts.append("[ASSISTANT]\n")
    return "".join(parts)

# ---------- inference ----------
def run_model(model_key: str, messages: List[Dict[str, str]]) -> str:
    llm = get_model(model_key)
    s = _settings_for(model_key)

    # Determine context + remaining space
    # n_ctx is what we initialized the model with
    prompt_token_count = None
    n_ctx = None
    try:
        prompt_str = format_messages(messages)
        prompt_tokens = llm.tokenize(prompt_str.encode("utf-8"))
        prompt_token_count = len(prompt_tokens)
        # match the constructor value (we don't have a public getter)
        n_ctx = int(os.getenv("LLM_N_CTX", str(s.get("n_ctx", 4096))))
        remaining_ctx = max(256, n_ctx - prompt_token_count - SAFETY_MARGIN)
    except Exception:
        remaining_ctx = 1024

    # max_tokens priority: env > config > remaining_ctx
    max_tokens_cfg = int(s.get("max_tokens")) if "max_tokens" in s else None
    max_tokens_env = os.getenv("LLM_MAX_TOKENS")
    if max_tokens_env is not None:
        try:
            max_tokens = int(max_tokens_env)
        except ValueError:
            max_tokens = None
    else:
        max_tokens = max_tokens_cfg

    # always cap by remaining_ctx to avoid overflow
    if max_tokens is None:
        max_tokens_final = remaining_ctx
    else:
        max_tokens_final = max(1, min(max_tokens, remaining_ctx))

    # Log prompt-side usage before calling the model
    if prompt_token_count is not None:
        logging.info(
            f"[tokens] model={model_key} prompt_tokens={prompt_token_count} "
            f"n_ctx={n_ctx if n_ctx is not None else 'unknown'} "
            f"max_gen_tokens={max_tokens_final}"
        )

    request_payload = {
        "messages": messages,
        "temperature": float(os.getenv("LLM_TEMP", str(s.get("temperature", 0.7)))),
        "top_p": float(os.getenv("LLM_TOP_P", str(s.get("top_p", 0.95)))),
        "top_k": int(os.getenv("LLM_TOP_K", str(s.get("top_k", 40)))),
        "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", str(s.get("repeat_penalty", 1.1)))),
        "mirostat_mode": int(os.getenv("LLM_MIROS", str(s.get("mirostat_mode", 0)))),
        "mirostat_tau": float(os.getenv("LLM_MIROS_TAU", str(s.get("mirostat_tau", 5.0)))),
        "mirostat_eta": float(os.getenv("LLM_MIROS_ETA", str(s.get("mirostat_eta", 0.1)))),
        "max_tokens": max_tokens_final,
    }

    logging.info("[debug] full request payload (truncated): %s",
                 json.dumps(request_payload, indent=2)[:1000])

    response = llm.create_chat_completion(
        messages=messages,
        temperature=request_payload["temperature"],
        top_p=request_payload["top_p"],
        top_k=request_payload["top_k"],
        repeat_penalty=request_payload["repeat_penalty"],
        presence_penalty=float(os.getenv("LLM_PRESENCE_PENALTY", str(s.get("presence_penalty", 0.0)))),
        frequency_penalty=float(os.getenv("LLM_FREQUENCY_PENALTY", str(s.get("frequency_penalty", 0.0)))),
        mirostat_mode=request_payload["mirostat_mode"],
        mirostat_tau=request_payload["mirostat_tau"],
        mirostat_eta=request_payload["mirostat_eta"],
        max_tokens=request_payload["max_tokens"],
    )

    # Different builds return slightly different shapes
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content")
    if text is None:
        text = choice.get("text", "")
    out = (text or "").strip()

    # Try to get usage from llama.cpp/llama-cpp-python; otherwise approximate
    usage = response.get("usage") or {}
    comp_tokens = usage.get("completion_tokens")
    prmpt_tokens_from_usage = usage.get("prompt_tokens", prompt_token_count)

    if comp_tokens is None:
        try:
            comp_tokens = len(llm.tokenize(out.encode("utf-8")))
        except Exception:
            comp_tokens = -1

    total_tokens = None
    try:
        total_tokens = (prmpt_tokens_from_usage or 0) + (comp_tokens or 0)
    except Exception:
        pass

    logging.info(
        f"[tokens] model={model_key} "
        f"prompt_tokens={prmpt_tokens_from_usage if prmpt_tokens_from_usage is not None else 'unknown'} "
        f"completion_tokens={comp_tokens if comp_tokens is not None else 'unknown'} "
        f"total_tokens={total_tokens if total_tokens is not None else 'unknown'}"
    )

    return out
