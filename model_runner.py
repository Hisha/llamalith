import os
import json
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

def _rope_kwargs(s: dict) -> dict:
    """
    Build rope-related kwargs for llama_cpp.Llama from config "rope_scaling".
    For llama-cpp-python 0.3.15, rope_scaling_type expects an *int* enum:
      0 = none, 1 = linear, 2 = yarn
    """
    out = {}
    rope = (s or {}).get("rope_scaling")
    if not rope:
        return out

    # Map string -> enum int expected by your wheel
    type_map = {"none": 0, "linear": 1, "yarn": 2}
    rtype_str = str(rope.get("type", "none")).lower()
    rtype_enum = type_map.get(rtype_str, 0)  # default to none

    # scale factor (e.g., 2.0 to go 4k->8k)
    factor = float(rope.get("factor", 1.0))

    # Required / common knobs
    if rtype_enum > 0:
        out["rope_scaling_type"] = rtype_enum
        out["rope_freq_scale"] = factor

    # YARN-specific good defaults if not provided
    # (these keys exist in your wheel signature)
    if rtype_enum == 2:  # yarn
        out.setdefault("yarn_orig_ctx", int(rope.get("yarn_orig_ctx", 4096)))
        out.setdefault("yarn_ext_factor", float(rope.get("yarn_ext_factor", -1.0)))
        out.setdefault("yarn_attn_factor", float(rope.get("yarn_attn_factor", 1.0)))
        out.setdefault("yarn_beta_fast", float(rope.get("yarn_beta_fast", 32.0)))
        out.setdefault("yarn_beta_slow", float(rope.get("yarn_beta_slow", 1.0)))

    return out

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

    # per-model / env overrides
    n_ctx = int(os.getenv("LLM_N_CTX", str(s.get("n_ctx", 4096))))
    n_threads = int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 8)))
    n_batch = int(os.getenv("LLM_N_BATCH", str(s.get("n_batch", 512))))
    n_gpu_layers = int(os.getenv("LLM_N_GPU_LAYERS", str(s.get("n_gpu_layers", 0))))

    # chat_format: env overrides config, which overrides fallback
    model_format = os.getenv("LLM_CHAT_FORMAT") or MODEL_FORMATS.get(model_key) or "llama-2"

    # gather optional RoPE scaling kwargs from config (only present for models you enable)
    rope_kwargs = _rope_kwargs(s)

    # Build common kwargs
    base_kwargs = dict(
        model_path=path,
        chat_format=model_format,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=n_batch,
        n_gpu_layers=n_gpu_layers,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )

    # Try with rope; if the installed llama-cpp-python is older and rejects args, retry without
    try:
        llm = Llama(**base_kwargs, **rope_kwargs)
        loaded_with_rope = bool(rope_kwargs)
    except TypeError:
        # Fall back gracefully (old wheel)
        llm = Llama(**base_kwargs)
        loaded_with_rope = False
        if rope_kwargs:
            print(f"[llamalith] WARNING: rope kwargs unsupported by current llama-cpp-python wheel; "
                  f"loaded without RoPE scaling for model={model_key}")

    _LOADED[model_key] = llm

    # Log a concise summary
    rope_msg = f" rope={rope_kwargs}" if loaded_with_rope else ""
    print(f"[llamalith] loaded model={model_key} path={path} "
          f"format={model_format} n_ctx={n_ctx} n_batch={n_batch} n_gpu_layers={n_gpu_layers}{rope_msg}")
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
    try:
        prompt_str = format_messages(messages)
        prompt_tokens = llm.tokenize(prompt_str.encode("utf-8"))
        # match the constructor value (we don't have a public getter)
        n_ctx = int(os.getenv("LLM_N_CTX", str(s.get("n_ctx", 4096))))
        remaining_ctx = max(256, n_ctx - len(prompt_tokens) - SAFETY_MARGIN)
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

    response = llm.create_chat_completion(
        messages=messages,
        temperature=float(os.getenv("LLM_TEMP", str(s.get("temperature", 0.7)))),
        top_p=float(os.getenv("LLM_TOP_P", str(s.get("top_p", 0.95)))),
        top_k=int(os.getenv("LLM_TOP_K", str(s.get("top_k", 40)))),
        repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", str(s.get("repeat_penalty", 1.1)))),
        mirostat_mode=int(os.getenv("LLM_MIROS", str(s.get("mirostat_mode", 0)))),   # 0/1/2
        mirostat_tau=float(os.getenv("LLM_MIROS_TAU", str(s.get("mirostat_tau", 5.0)))),
        mirostat_eta=float(os.getenv("LLM_MIROS_ETA", str(s.get("mirostat_eta", 0.1)))),
        max_tokens=max_tokens_final,
    )

    # Different builds return slightly different shapes
    choice = (response.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content")
    if text is None:
        text = choice.get("text", "")
    return (text or "").strip()
