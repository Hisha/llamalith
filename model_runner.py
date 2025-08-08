import os
import json
from typing import List, Dict
from llama_cpp import Llama

SAFETY_MARGIN = int(os.getenv("LLM_SAFETY_MARGIN", "128"))

# ---------- config ----------
CONFIG_PATH = os.getenv("LLAMALITH_CONFIG", "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        MODEL_PATHS = config.get("model_paths", {})
else:
    MODEL_PATHS = {
        "mistral": os.getenv("MISTRAL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "mythomax": os.getenv("MYTHOMAX_PATH", "models/mythomax-l2-13b.Q4_K_M.gguf"),
    }

# ---------- model cache ----------
_LOADED: Dict[str, Llama] = {}

def get_model(model_key: str) -> Llama:
    """Singleton loader for each model key."""
    if model_key in _LOADED:
        return _LOADED[model_key]
    path = MODEL_PATHS.get(model_key)
    if not path or not os.path.exists(path):
        raise ValueError(f"Model '{model_key}' not found or path does not exist: {path!r}")
    llm = Llama(
        model_path=path,
        n_ctx=4096,                     # you can raise this if you build with larger context
        n_threads=os.cpu_count() or 8,  # use all cores
        n_batch=512,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    _LOADED[model_key] = llm
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

    # Estimate how many tokens are left in context
    # (tokenize a simple stringified view of the chat)
    try:
        prompt_str = format_messages(messages)
        prompt_tokens = llm.tokenize(prompt_str.encode("utf-8"))
        n_ctx = 4096  # matches what we initialized with
        remaining = max(256, n_ctx - len(prompt_tokens) - SAFETY_MARGIN)
    except Exception:
        # Fallback if tokenize differs in your build
        remaining = 1024

    response = llm.create_chat_completion(
        messages=messages,
        temperature=float(os.getenv("LLM_TEMP", "0.7")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        top_k=int(os.getenv("LLM_TOP_K", "40")),
        repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.1")),
        mirostat_mode=int(os.getenv("LLM_MIROS", "0")),   # 0/1/2
        mirostat_tau=float(os.getenv("LLM_MIROS_TAU", "5.0")),
        mirostat_eta=float(os.getenv("LLM_MIROS_ETA", "0.1")),
        # stops mainly matter if someone uses a plain prompt format;
        # for chat format they’re usually ignored by llama.cpp
        stop=["</s>", "\nUser:", "\nuser:", "\n###", "\nAssistant:"],
        max_tokens=remaining,   # elastic; not a hard UX cap
    )

    # Different builds return slightly different shapes
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content")
    if text is None:
        text = choice.get("text", "")
    return (text or "").strip()
