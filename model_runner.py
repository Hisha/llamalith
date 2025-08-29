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

    # ---- context accounting ----
    prompt_token_count = None
    n_ctx = int(os.getenv("LLM_N_CTX", str(s.get("n_ctx", 4096))))
    try:
        prompt_str = format_messages(messages)
        prompt_tokens = llm.tokenize(prompt_str.encode("utf-8"))
        prompt_token_count = len(prompt_tokens)
        remaining_ctx = max(256, n_ctx - prompt_token_count - SAFETY_MARGIN)
    except Exception:
        remaining_ctx = 1024

    # ---- max tokens (env > config > remaining_ctx) ---
    max_tokens_cfg = s.get("max_tokens")
    max_tokens_env = os.getenv("LLM_MAX_TOKENS")
    max_tokens = None
    if max_tokens_env is not None:
        try:
            max_tokens = int(max_tokens_env)
        except ValueError:
            max_tokens = None
    elif isinstance(max_tokens_cfg, int):
        max_tokens = max_tokens_cfg
    max_tokens_final = max(1, min(max_tokens or remaining_ctx, remaining_ctx))

    # ---- sampling/decoding params (env overlaying config defaults) ----
    params = {
        "temperature": float(os.getenv("LLM_TEMP", str(s.get("temperature", 0.8)))),
        "top_p": float(os.getenv("LLM_TOP_P", str(s.get("top_p", 0.9)))),
        "top_k": int(os.getenv("LLM_TOP_K", str(s.get("top_k", 40)))),
        "repeat_penalty": float(os.getenv("LLM_REPEAT_PENALTY", str(s.get("repeat_penalty", 1.07)))),
        "max_tokens": max_tokens_final,
        # removed "ignore_eos": True (backend rejects this kwarg)
    }

    # optional typical sampling
    try:
        typ = os.getenv("LLM_TYPICAL_P", str(s.get("typical_p", 0.97)))
        if typ is not None:
            params["typical_p"] = float(typ)
    except Exception:
        pass

    # presence/frequency penalties: default to 0.0 for prose
    def _float_or_none(v):
        try:
            return float(v)
        except Exception:
            return None

    pres = os.getenv("LLM_PRESENCE_PENALTY", None)
    if pres is None and "presence_penalty" in s:
        pres = s.get("presence_penalty")
    pres_val = _float_or_none(pres)
    if pres_val is None:
        pres_val = 0.0
    params["presence_penalty"] = pres_val

    freq = os.getenv("LLM_FREQUENCY_PENALTY", None)
    if freq is None and "frequency_penalty" in s:
        freq = s.get("frequency_penalty")
    freq_val = _float_or_none(freq)
    if freq_val is None:
        freq_val = 0.0
    params["frequency_penalty"] = freq_val

    # Stop sequences: disable for prose model to avoid early cutoffs
    stop = None
    stop_env = os.getenv("LLM_STOP")
    stop_cfg = s.get("stop")
    if not model_key.endswith("-novelchapter"):
        if stop_env:
            stop = [x for x in stop_env.split(",") if x]
        elif isinstance(stop_cfg, list):
            stop = stop_cfg
        elif isinstance(stop_cfg, str):
            stop = [stop_cfg]
    if stop:
        params["stop"] = stop

    # ---- Logging: tokens + param snapshot (no full prompts) ----
    if prompt_token_count is not None:
        logging.info(
            f"[tokens] model={model_key} prompt_tokens={prompt_token_count} "
            f"n_ctx={n_ctx} max_gen_tokens={max_tokens_final}"
        )
    try:
        msg_sizes = [len((m.get("content") or "").encode("utf-8")) for m in messages]
        logging.info(
            "[request] model=%s temp=%.3f top_p=%.3f top_k=%d rep=%.3f typ=%.3f "
            "pres=%.3f freq=%.3f max_tokens=%d stop=%s messages=%d bytes=%d",
            model_key,
            params["temperature"], params["top_p"], params["top_k"], params["repeat_penalty"],
            params.get("typical_p", -1.0),
            params["presence_penalty"], params["frequency_penalty"],
            params["max_tokens"], params.get("stop", "-"),
            len(messages), sum(msg_sizes),
        )
        roles = [m.get("role","") for m in messages]
        logging.info("[request] message_roles=%s", roles)
    except Exception as e:
        logging.warning(f"[request] failed to log param snapshot: {e}")

    # ---- Call model ----
    response = llm.create_chat_completion(messages=messages, **params)

    # ---- Extract text + finish reason ----
    choice = (response.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or response.get("finish_reason")
    if finish:
        logging.info("[response] finish_reason=%s", finish)

    msg = choice.get("message", {})
    text = msg.get("content")
    if text is None:
        text = choice.get("text", "")
    out = (text or "").strip()

    # ---- Usage logging ----
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
        f"[tokens] model={model_key} prompt_tokens={prmpt_tokens_from_usage if prmpt_tokens_from_usage is not None else 'unknown'} "
        f"completion_tokens={comp_tokens if comp_tokens is not None else 'unknown'} "
        f"total_tokens={total_tokens if total_tokens is not None else 'unknown'}"
    )

    return out
