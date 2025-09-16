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
def run_model(model_key: str, messages: List[Dict[str, str]], grammar_name: str = None) -> str:
    import os, logging

    llm = get_model(model_key)
    s = _settings_for(model_key)

    # --- Grammar file option ---
    grammar_text = None
    if grammar_name:
        # sanitize: allow only safe filename chars
        safe = "".join(ch for ch in grammar_name if ch.isalnum() or ch in ("-", "_", ".", "+"))
        # drop path components
        safe = os.path.basename(safe)
        if not safe.endswith(".gbnf"):
            safe = safe + ".gbnf"
        grammar_dir = os.getenv("LLM_GRAMMAR_DIR", "/home/smithkt/llama.cpp/grammars")
        grammar_path = os.path.join(grammar_dir, safe)
        try:
            with open(grammar_path, "r", encoding="utf-8") as gf:
                grammar_text = gf.read()
                logging.info("[grammar] using %s", grammar_path)
        except Exception as e:
            logging.warning("[grammar] failed to load %s: %s", grammar_path, e)
    
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

    # ---- end token + stop handling ----
    end_token_str = s.get("end_token", "<<END>>")
    stop = None
    stop_env = os.getenv("LLM_STOP")
    stop_cfg = s.get("stop")
    require_end = bool(s.get("require_end_token"))  # per-model toggle

    if require_end:
        stop = [end_token_str]  # enforce literal end marker
    elif not model_key.endswith("-novelchapter"):
        if stop_env:
            stop = [x for x in stop_env.split(",") if x]
        elif isinstance(stop_cfg, list):
            stop = stop_cfg
        elif isinstance(stop_cfg, str):
            stop = [stop_cfg]
    if stop:
        params["stop"] = stop

    # ---- EOS + END-token logit bias (adapter-key with fallbacks) ----
    eos_bias_value = None
    try:
        eos_bias_val = os.getenv("LLM_EOS_BIAS", None)
        if eos_bias_val is None:
            eos_bias_val = s.get("eos_bias", None)  # per-model config
        if eos_bias_val is not None:
            eos_bias_value = float(eos_bias_val)
    except Exception:
        eos_bias_value = None

    # end-token positive bias map (and EOS negative)
    bias_map = None
    end_token_ids = []
    try:
        if eos_bias_value is not None or require_end:
            bias_map = {}
            # EOS discourage (llama.cpp EOS = 2)
            if eos_bias_value is not None:
                bias_map[2] = eos_bias_value
            # Encourage literal end marker tokens
            if require_end:
                try:
                    end_token_ids = llm.tokenize(end_token_str.encode("utf-8"))
                    for tid in end_token_ids:
                        # gentle positive bias per sub-token of the marker
                        bias_map[tid] = bias_map.get(tid, 0.0) + 8.0
                except Exception:
                    end_token_ids = []
    except Exception:
        bias_map = None

    if grammar_text:
        params.pop("stop", None)     # grammar governs termination
        require_end = False          # skip manual continuation loop
        params["grammar"] = grammar_text
    
    # choose adapter key for logit bias (config/env + fallbacks)
    candidate_bias_keys = []
    pref_key = os.getenv("LLM_LOGIT_BIAS_KEY", s.get("logit_bias_key", None))
    if pref_key:
        candidate_bias_keys.append(pref_key)
    for k in ("logit_bias", "logit-bias", "logit_bias_map"):
        if k not in candidate_bias_keys:
            candidate_bias_keys.append(k)

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
    except Exception as e:
        logging.warning(f"[request] failed to log param snapshot: {e}")

    # ---- Call model (with progressive fallback for bias kwarg) ----
    bias_key_used = None

    def _try_call(with_bias_key: str = None, given_params=None, given_messages=None):
        call_params = dict(given_params or params)
        if with_bias_key and bias_map:
            call_params[with_bias_key] = bias_map
        return llm.create_chat_completion(messages=(given_messages or messages), **call_params)

    response = None
    if bias_map:
        for key in candidate_bias_keys:
            try:
                response = _try_call(key)
                bias_key_used = key
                break
            except TypeError as te:
                if f"unexpected keyword argument '{key}'" in str(te):
                    continue
                raise
            except Exception:
                continue

    if response is None:
        response = _try_call(None)
        logging.info("[request] eos_bias=end-token-bias=%s",
                     "none" if not bias_map else "adapter-ignored")

    if bias_key_used:
        logging.info("[request] eos_bias_applied=%s=%s; end_token_ids=%s",
                     bias_key_used, {2: eos_bias_value} if eos_bias_value is not None else {},
                     end_token_ids if end_token_ids else "[]")

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

    # ---- Require-end enforcement (multi-continue, capped) ----
    # max_continues: env -> per-model -> default 1
    def _int_or(default_v: int, v):
        try:
            return int(v)
        except Exception:
            return default_v

    max_continues = _int_or(1, os.getenv("STORY_MAX_CONTINUES", s.get("max_continues", 1)))
    continues = 0

    def _headroom(current_out: str) -> int:
        try:
            out_tok = len(llm.tokenize(current_out.encode("utf-8")))
            return max(128, n_ctx - (prompt_token_count or 0) - out_tok - SAFETY_MARGIN)
        except Exception:
            return max(128, remaining_ctx // 2)

    while require_end and (end_token_str not in out) and (continues < max_continues):
        continues += 1
        logging.warning("[require_end_token] '%s' not found; continuation attempt %d/%d",
                        end_token_str, continues, max_continues)

        cont_messages = list(messages) + [
            {"role": "assistant", "content": out},
            {"role": "user", "content":
                f"Continue the same scene seamlessly without restarting. "
                f"When complete, end with {end_token_str} on its own line. Prose only."}
        ]
        headroom = _headroom(out)
        cont_params = dict(params)
        cont_params["max_tokens"] = max(256, min(int(params.get("max_tokens", 1024) * 0.6), headroom))

        # carry through the same bias key if we had one
        if bias_key_used and bias_map:
            cont_call_params = dict(cont_params)
            cont_call_params[bias_key_used] = bias_map
            cont_resp = llm.create_chat_completion(messages=cont_messages, **cont_call_params)
        else:
            cont_resp = llm.create_chat_completion(messages=cont_messages, **cont_params)

        cont_choice = (cont_resp.get("choices") or [{}])[0]
        cont_msg = cont_choice.get("message", {})
        cont_text = cont_msg.get("content") or cont_choice.get("text", "") or ""
        addition = cont_text.strip()
        if addition:
            out = (out + ("\n\n" if not out.endswith("\n") else "") + addition).strip()

        if end_token_str in out:
            break

    # Final failsafe: append end marker so downstream never hangs
    if require_end and end_token_str not in out:
        logging.warning("[require_end_token] still missing after %d attempt(s); appending %s",
                        continues, end_token_str)
        out = out.rstrip() + ("\n" if not out.endswith("\n") else "") + end_token_str

    logging.info("reply_len=%d", len(out))
    return out
