from llama_cpp import Llama
import os

# Path to GGUF models (can be overridden by env or config)
MODEL_PATHS = {
    "mistral": os.getenv("MISTRAL_PATH", "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"),
    "mythomax": os.getenv("MYTHOMAX_PATH", "models/mythomax-l2-13b.Q4_K_M.gguf")
}

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
            n_threads=8,
            use_mlock=True
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
def run_model(model_key, messages):
    model = get_model(model_key)
    prompt = format_messages(messages)

    output = model(
        prompt,
        max_tokens=512,
        stop=["[USER]", "[SYSTEM]", "[ASSISTANT]"],
        echo=False
    )

    return output["choices"][0]["text"].strip()
