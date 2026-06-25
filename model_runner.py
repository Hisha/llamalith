import os
import json
import gc
import logging
from typing import List, Dict, Any, Optional

from llama_cpp import Llama, LlamaGrammar

SAFETY_MARGIN = int(os.getenv("LLM_SAFETY_MARGIN", "128"))

# ---------- config ----------
CONFIG_PATH = os.getenv("LLAMALITH_CONFIG", "config.json")

AVAILABLE_MODELS: List[str] = []
MODEL_PATHS: Dict[str, str] = {}
MODEL_FORMATS: Dict[str, str] = {}
MODEL_SETTINGS: Dict[str, Dict[str, Any]] = {}

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    MODEL_PATHS = cfg.get("model_paths", {}) or {}
    MODEL_FORMATS = cfg.get("model_formats", {}) or {}
    MODEL_SETTINGS = cfg.get("model_settings", {}) or {}
    AVAILABLE_MODELS = cfg.get("available_models", []) or sorted(MODEL_PATHS.keys())
else:
    AVAILABLE_MODELS = ["gemma4", "mistral", "mythomax", "openchat", "qwen3"]

    MODEL_PATHS = {
        "gemma4": os.getenv(
            "GEMMA4_PATH",
            "models/gemma-4-e4b/gemma-4-E4B-it-Q4_K_M.gguf",
        ),
        "mistral": os.getenv(
            "MISTRAL_PATH",
            "models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        ),
        "mythomax": os.getenv(
            "MYTHOMAX_PATH",
            "models/mythomax/mythomax-l2-13b.Q4_K_M.gguf",
        ),
        "openchat": os.getenv(
            "OPENCHAT_PATH",
            "models/openchat/openchat-3.5-1210.Q8_0.gguf",
        ),
        "qwen3": os.getenv(
			"QWEN3_PATH",
			"models/qwen3/Qwen3-8B-Q4_K_M.gguf",
		),
    }

    MODEL_FORMATS = {
        "gemma4": "auto",
        "mistral": "mistral-instruct",
        "mythomax": "chatml",
        "openchat": "chatml",
        "qwen3": "auto",
    }

    MODEL_SETTINGS = {}

print(f"[llamalith] CONFIG_PATH={CONFIG_PATH}")
print(f"[llamalith] available_models={AVAILABLE_MODELS}")
print(f"[llamalith] model_paths={MODEL_PATHS}")
print(f"[llamalith] model_formats={MODEL_FORMATS}")

# ---------- model cache ----------
# Keep only ONE model loaded at a time to avoid CPU/RAM exhaustion.
_LOADED: Dict[str, Llama] = {}


def _settings_for(model_key: str) -> Dict[str, Any]:
    return MODEL_SETTINGS.get(model_key, {}) or {}


def _unload_all_models() -> None:
    global _LOADED

    if _LOADED:
        print(f"[llamalith] unloading cached models: {list(_LOADED.keys())}")

    _LOADED.clear()
    gc.collect()


def get_model(model_key: str) -> Llama:
    """Load a model by key. Keeps only one model resident at a time."""

    if model_key in _LOADED:
        return _LOADED[model_key]

    # Important: avoid keeping mistral/mythomax/openchat/gemma4 all in RAM.
    _unload_all_models()

    path = MODEL_PATHS.get(model_key)

    if not path or not os.path.exists(path):
        raise ValueError(
            f"Model '{model_key}' not found or path does not exist: {path!r}"
        )

    s = _settings_for(model_key)

    n_ctx = int(os.getenv("LLM_N_CTX", str(s.get("n_ctx", 4096))))

    model_format = (
        os.getenv("LLM_CHAT_FORMAT")
        or MODEL_FORMATS.get(model_key)
        or "auto"
    )

    llama_kwargs = {
        "model_path": path,
        "n_ctx": n_ctx,
        "n_threads": int(os.getenv("LLM_N_THREADS", str(os.cpu_count() or 8))),
        "n_batch": int(os.getenv("LLM_N_BATCH", "512")),
        "use_mmap": True,
        "use_mlock": False,
        "verbose": bool(int(os.getenv("LLM_VERBOSE", "0"))),
    }

    # For Gemma 4 / modern GGUFs, let llama-cpp-python use tokenizer.chat_template.
    # Explicit chat_format is only needed for older models.
    if model_format and model_format not in ("auto", "chat_template"):
        llama_kwargs["chat_format"] = model_format

    print(
        f"[llamalith] loading model={model_key} "
        f"path={path} "
        f"model_format={model_format} "
        f"n_ctx={n_ctx}"
    )

    llm = Llama(**llama_kwargs)

    _LOADED[model_key] = llm

    print(f"[llamalith] loaded model={model_key}")

    return llm


# ---------- prompt helpers ----------
def format_messages(messages: List[Dict[str, str]]) -> str:
    """Plain string representation, used only for token counting."""

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


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _int_or(default_value: int, value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return default_value


# ---------- inference ----------
def run_model(
    model_key: str,
    messages: List[Dict[str, str]],
    grammar_name: str = None,
) -> str:
    llm = get_model(model_key)
    s = _settings_for(model_key)

    # ---- Grammar file option ----
    grammar_text = None
    grammar_path = None

    if grammar_name:
        safe = "".join(
            ch for ch in grammar_name
            if ch.isalnum() or ch in ("-", "_", ".", "+")
        )
        safe = os.path.basename(safe)

        if not safe.endswith(".gbnf"):
            safe += ".gbnf"

        grammar_dir = os.getenv(
            "LLM_GRAMMAR_DIR",
            "/home/smithkt/llama.cpp/grammars",
        )

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

    # ---- max tokens ----
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

    # ---- sampling/decoding params ----
    params = {
        "temperature": float(os.getenv("LLM_TEMP", str(s.get("temperature", 0.8)))),
        "top_p": float(os.getenv("LLM_TOP_P", str(s.get("top_p", 0.9)))),
        "top_k": int(os.getenv("LLM_TOP_K", str(s.get("top_k", 40)))),
        "repeat_penalty": float(
            os.getenv("LLM_REPEAT_PENALTY", str(s.get("repeat_penalty", 1.07)))
        ),
        "max_tokens": max_tokens_final,
    }

    # optional typical_p
    try:
        typical_p = os.getenv("LLM_TYPICAL_P", str(s.get("typical_p", 0.97)))
        if typical_p is not None:
            params["typical_p"] = float(typical_p)
    except Exception:
        pass

    # presence/frequency penalties
    pres = os.getenv("LLM_PRESENCE_PENALTY", None)
    if pres is None and "presence_penalty" in s:
        pres = s.get("presence_penalty")

    pres_val = _float_or_none(pres)
    params["presence_penalty"] = 0.0 if pres_val is None else pres_val

    freq = os.getenv("LLM_FREQUENCY_PENALTY", None)
    if freq is None and "frequency_penalty" in s:
        freq = s.get("frequency_penalty")

    freq_val = _float_or_none(freq)
    params["frequency_penalty"] = 0.0 if freq_val is None else freq_val

    # ---- end token + stop handling ----
    end_token_str = s.get("end_token", "<<END>>")
    stop = None
    stop_env = os.getenv("LLM_STOP")
    stop_cfg = s.get("stop")
    require_end = bool(s.get("require_end_token"))

    if require_end:
        stop = [end_token_str]
    elif not model_key.endswith("-novelchapter"):
        if stop_env:
            stop = [x for x in stop_env.split(",") if x]
        elif isinstance(stop_cfg, list):
            stop = stop_cfg
        elif isinstance(stop_cfg, str):
            stop = [stop_cfg]

    if stop:
        params["stop"] = stop

    # ---- EOS + END-token logit bias ----
    eos_bias_value = None

    try:
        eos_bias_val = os.getenv("LLM_EOS_BIAS", None)

        if eos_bias_val is None:
            eos_bias_val = s.get("eos_bias", None)

        if eos_bias_val is not None:
            eos_bias_value = float(eos_bias_val)
    except Exception:
        eos_bias_value = None

    bias_map = None
    end_token_ids = []

    try:
        if eos_bias_value is not None or require_end:
            bias_map = {}

            if eos_bias_value is not None:
                bias_map[2] = eos_bias_value

            if require_end:
                try:
                    end_token_ids = llm.tokenize(end_token_str.encode("utf-8"))
                    for tid in end_token_ids:
                        bias_map[tid] = bias_map.get(tid, 0.0) + 8.0
                except Exception:
                    end_token_ids = []
    except Exception:
        bias_map = None

    # ---- grammar ----
    if grammar_text:
        params.pop("stop", None)
        require_end = False

        grammar_obj = None

        try:
            grammar_obj = LlamaGrammar.from_string(grammar_text, "root")
        except TypeError:
            try:
                grammar_obj = LlamaGrammar.from_string(grammar_text)
            except Exception as e1:
                try:
                    grammar_obj = LlamaGrammar.from_file(grammar_path)
                except Exception as e2:
                    logging.exception(
                        "[grammar] failed to construct LlamaGrammar: %s / %s",
                        e1,
                        e2,
                    )

        if grammar_obj is not None:
            params["grammar"] = grammar_obj
            logging.info("[grammar] attached OK")
        else:
            logging.warning("[grammar] disabled due to construction error")

    # ---- logit bias adapter key ----
    candidate_bias_keys = []

    pref_key = os.getenv("LLM_LOGIT_BIAS_KEY", s.get("logit_bias_key", None))

    if pref_key:
        candidate_bias_keys.append(pref_key)

    for key in ("logit_bias", "logit-bias", "logit_bias_map"):
        if key not in candidate_bias_keys:
            candidate_bias_keys.append(key)

    # ---- Logging ----
    if prompt_token_count is not None:
        logging.info(
            "[tokens] model=%s prompt_tokens=%s n_ctx=%s max_gen_tokens=%s",
            model_key,
            prompt_token_count,
            n_ctx,
            max_tokens_final,
        )

    try:
        msg_sizes = [len((m.get("content") or "").encode("utf-8")) for m in messages]

        logging.info(
            "[request] model=%s temp=%.3f top_p=%.3f top_k=%d rep=%.3f "
            "typ=%.3f pres=%.3f freq=%.3f max_tokens=%d stop=%s messages=%d bytes=%d",
            model_key,
            params["temperature"],
            params["top_p"],
            params["top_k"],
            params["repeat_penalty"],
            params.get("typical_p", -1.0),
            params["presence_penalty"],
            params["frequency_penalty"],
            params["max_tokens"],
            params.get("stop", "-"),
            len(messages),
            sum(msg_sizes),
        )
    except Exception as e:
        logging.warning("[request] failed to log param snapshot: %s", e)

    # ---- call model ----
    bias_key_used = None

    def _try_call(
        with_bias_key: str = None,
        given_params: Dict[str, Any] = None,
        given_messages: List[Dict[str, str]] = None,
    ):
        call_params = dict(given_params or params)

        if with_bias_key and bias_map:
            call_params[with_bias_key] = bias_map

        return llm.create_chat_completion(
            messages=(given_messages or messages),
            **call_params,
        )

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

        logging.info(
            "[request] eos_bias=end-token-bias=%s",
            "none" if not bias_map else "adapter-ignored",
        )

    if bias_key_used:
        logging.info(
            "[request] eos_bias_applied=%s=%s; end_token_ids=%s",
            bias_key_used,
            {2: eos_bias_value} if eos_bias_value is not None else {},
            end_token_ids if end_token_ids else "[]",
        )

    # ---- extract text ----
    choice = (response.get("choices") or [{}])[0]
    finish = choice.get("finish_reason") or response.get("finish_reason")

    if finish:
        logging.info("[response] finish_reason=%s", finish)

    msg = choice.get("message", {})
    text = msg.get("content")

    if text is None:
        text = choice.get("text", "")

    out = (text or "").strip()

    # ---- usage logging ----
    usage = response.get("usage") or {}
    comp_tokens = usage.get("completion_tokens")
    prompt_tokens_from_usage = usage.get("prompt_tokens", prompt_token_count)

    if comp_tokens is None:
        try:
            comp_tokens = len(llm.tokenize(out.encode("utf-8")))
        except Exception:
            comp_tokens = -1

    try:
        total_tokens = (prompt_tokens_from_usage or 0) + (comp_tokens or 0)
    except Exception:
        total_tokens = None

    logging.info(
        "[tokens] model=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
        model_key,
        prompt_tokens_from_usage if prompt_tokens_from_usage is not None else "unknown",
        comp_tokens if comp_tokens is not None else "unknown",
        total_tokens if total_tokens is not None else "unknown",
    )

    # ---- Require-end enforcement ----
    max_continues = _int_or(
        1,
        os.getenv("STORY_MAX_CONTINUES", s.get("max_continues", 1)),
    )

    continues = 0

    def _headroom(current_out: str) -> int:
        try:
            out_tok = len(llm.tokenize(current_out.encode("utf-8")))
            return max(128, n_ctx - (prompt_token_count or 0) - out_tok - SAFETY_MARGIN)
        except Exception:
            return max(128, remaining_ctx // 2)

    while require_end and (end_token_str not in out) and (continues < max_continues):
        continues += 1

        logging.warning(
            "[require_end_token] '%s' not found; continuation attempt %d/%d",
            end_token_str,
            continues,
            max_continues,
        )

        cont_messages = list(messages) + [
            {"role": "assistant", "content": out},
            {
                "role": "user",
                "content": (
                    "Continue the same scene seamlessly without restarting. "
                    f"When complete, end with {end_token_str} on its own line. Prose only."
                ),
            },
        ]

        headroom = _headroom(out)
        cont_params = dict(params)
        cont_params["max_tokens"] = max(
            256,
            min(int(params.get("max_tokens", 1024) * 0.6), headroom),
        )

        if bias_key_used and bias_map:
            cont_call_params = dict(cont_params)
            cont_call_params[bias_key_used] = bias_map

            cont_resp = llm.create_chat_completion(
                messages=cont_messages,
                **cont_call_params,
            )
        else:
            cont_resp = llm.create_chat_completion(
                messages=cont_messages,
                **cont_params,
            )

        cont_choice = (cont_resp.get("choices") or [{}])[0]
        cont_msg = cont_choice.get("message", {})
        cont_text = cont_msg.get("content") or cont_choice.get("text", "") or ""
        addition = cont_text.strip()

        if addition:
            out = (out + ("\n\n" if not out.endswith("\n") else "") + addition).strip()

        if end_token_str in out:
            break

    if require_end and end_token_str not in out:
        logging.warning(
            "[require_end_token] still missing after %d attempt(s); appending %s",
            continues,
            end_token_str,
        )

        out = out.rstrip() + ("\n" if not out.endswith("\n") else "") + end_token_str

    logging.info("reply_len=%d", len(out))

    return out