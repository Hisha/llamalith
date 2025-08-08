from llama_cpp import Llama
from typing import List, Dict, Optional
import os
import json

SAFETY_MARGIN = int(os.getenv("LLM_SAFETY_MARGIN", "128"))

# Load model paths from config file or environment
CONFIG_PATH = os.getenv("LLAMALITH_CONFIG", "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        MODEL_PATHS = config.get("model_paths", {})
        MAX_TOKENS = config.get("max_tokens", 512)
else:
    MODEL_PATHS = {
        "mistral": os.getenv("MISTRAL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
        "mythomax": os.getenv("MYTHOMAX_PATH", "models/mythomax-l2-13b.Q4_K_M.gguf")
    }
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 512))

# Cache loaded models
LOADED_MODELS = {}

# Load model if not already loaded
def get_model(model_key):
    if model_key not in LOADED_MODELS:
        path = MODEL_PATHS.get(model_key)
        if not path or not os.path.exists(path):
            raise ValueError(f"Model '{model_key}' not found or path does not exist.")

        LOADED_MODELS[model_key] = Llama(
            model_path=path,
            n_ctx=4096,
            n_threads=24,
            n_batch=512,
            use_mmap=True,
            use_mlock=False
        )
    return LOADED_MODELS[model_key]

# Convert messages to single prompt
def format_messages(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"[SYSTEM]\n{content}\n"
        elif role == "user":
            prompt += f"[USER]\n{content}\n"
        elif role == "assistant":
            prompt += f"[ASSISTANT]\n{content}\n"
    prompt += "[ASSISTANT]\n"
    return prompt

# Run a model with message history
def run_model(model_key: str, messages: List[Dict[str, str]]) -> str:
    llm = get_llama_instance(model_key)  # your loader with n_threads set (see below)
    n_ctx = getattr(llm, "n_ctx", lambda: 4096)()
    # build your chat prompt string or tokens
    prompt = format_chat(messages)         # your function
    prompt_tokens = llm.tokenize(prompt.encode("utf-8"))
    remaining = max(256, n_ctx - len(prompt_tokens) - SAFETY_MARGIN)  # elastic

    resp = llm.create_chat_completion(
        messages=messages,
        # sampling knobs via env if you like
        temperature=float(os.getenv("LLM_TEMP", "0.7")),
        top_p=float(os.getenv("LLM_TOP_P", "0.95")),
        top_k=int(os.getenv("LLM_TOP_K", "40")),
        repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.1")),
        mirostat_mode=int(os.getenv("LLM_MIROS", "0")),       # 0/1/2
        mirostat_tau=float(os.getenv("LLM_MIROS_TAU", "5.0")),
        mirostat_eta=float(os.getenv("LLM_MIROS_ETA", "0.1")),
        stop=["</s>", "\nUser:", "\nuser:", "\n###", "\nAssistant:"],
        max_tokens=remaining,
    )
    return resp["choices"][0]["message"]["content"].strip()
