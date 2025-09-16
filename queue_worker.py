# queue_worker.py
import multiprocessing
import time
import sys
import re
import logging, os, traceback

CONFIG_PATH = os.getenv("LLAMALITH_CONFIG", "config.json")
_story_cfg = {}
_worker_count = 2  # fallback default

if os.path.exists(CONFIG_PATH):
    try:
        import json
        with open(CONFIG_PATH, "r") as _f:
            _cfg = json.load(_f) or {}
            _story_cfg = (_cfg.get("story_settings") or {})
            _worker_count = int(_cfg.get("worker_settings", {}).get("worker_count", 2))
    except Exception:
        _story_cfg = {}
        _worker_count = 2

from memory import (
    claim_next_job,
    get_conversation_messages,
    save_assistant_message,
    mark_job_done,
)
from model_runner import run_model

POLL_SEC = 1
NUM_WORKERS = _worker_count

logging.basicConfig(
    level=logging.INFO,
    format='[Worker %(process)d] %(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ---- SSML helpers / knobs ----
TARGET_MIN_WORDS = int(os.getenv("STORY_MIN_WORDS", str(_story_cfg.get("min_words", 700))))
MAX_CONTINUES    = int(os.getenv("STORY_MAX_CONTINUES", str(_story_cfg.get("max_continues", 2))))

SSML_SPEAK_RE = re.compile(r"<\s*/?\s*speak\s*>", re.I)
SPEAK_OPEN_RE  = re.compile(r"<\s*speak\s*>", re.I)
SPEAK_CLOSE_RE = re.compile(r"</\s*speak\s*>", re.I)

def normalize_speak_once(text: str) -> str:
    """Remove any stray <speak> / </speak> anywhere, then wrap once."""
    if not text:
        return "<speak></speak>"
    body = SPEAK_OPEN_RE.sub("", text)
    body = SPEAK_CLOSE_RE.sub("", body)
    body = body.strip()
    return f"<speak>\n{body}\n</speak>"
    
def strip_ssml_tags(s: str) -> str:
    # remove <speak>, </speak>, and any other tags like <break .../>
    no_speak = SSML_SPEAK_RE.sub("", s or "")
    return re.sub(r"<[^>]+>", " ", no_speak).strip()

def word_count(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s or ""))

def extract_inner_ssml(s: str) -> str:
    m = re.search(r"<\s*speak\s*>(.*)</\s*speak\s*>", s or "", flags=re.S | re.I)
    return (m.group(1) if m else (s or "")).strip()

def wrap_speak(inner: str) -> str:
    inner = (inner or "").strip()
    return inner if inner.lower().startswith("<speak>") else f"<speak>\n{inner}\n</speak>"

def is_probably_ssml(system_prompt: str, text: str) -> bool:
    sp = (system_prompt or "").lower()
    t  = (text or "").lower()
    return ("ssml" in sp) or ("<speak" in t and "</speak" in t)

# ---- main worker ----
def worker_loop(worker_id: int):
    print(f"🚀 Worker {worker_id} started.", flush=True)
    while True:
        try:
            job = claim_next_job()
            if not job:
                time.sleep(POLL_SEC)
                continue

            jid = job["id"]
            convo_id = job["conversation_id"]
            model_key = job["model"]
            user_input = (job.get("user_input") or "").strip()
            system_prompt = (job.get("system_prompt") or "").strip()
            grammar_name = (job.get("grammar_name") or "").strip() or None

            logging.info(f"claimed job={jid} model={model_key} convo={convo_id} ulen={len(user_input)}")

            # Build history from DB
            history = get_conversation_messages(convo_id) or []

            # Ensure current user turn is present
            if user_input and (not history or history[-1].get("role") != "user" or history[-1].get("content","").strip() != user_input):
                history.append({"role": "user", "content": user_input})

            # Ensure system prompt at top (avoid dup)
            if system_prompt and (not history or history[0].get("role") != "system" or history[0].get("content","").strip() != system_prompt):
                history.insert(0, {"role": "system", "content": system_prompt})

            logging.info(f"history_turns={len(history)}; calling model…")
            reply = (run_model(model_key, history, grammar_name=grammar_name) or "").strip()
            logging.info(f"reply_len={len(reply)}")

            if not reply:
                mark_job_done(jid, failed=True, result_text="Empty model output")
                logging.warning(f"empty output -> marked job {jid} failed")
                continue

            # ---- SSML length guard (ONLY if this looks like SSML) ----
            if is_probably_ssml(system_prompt, reply):
                inner = extract_inner_ssml(reply)
                wc = word_count(strip_ssml_tags(inner))
                continues_left = MAX_CONTINUES

                # Build a local history that includes the assistant reply so far
                local_history = list(history)
                local_history.append({"role": "assistant", "content": wrap_speak(inner)})

                while wc < TARGET_MIN_WORDS and continues_left > 0:
                    need = TARGET_MIN_WORDS - wc
                    # Ask the model to continue the SAME story, no new <speak> wrapper
                    cont_prompt = (
                        "Continue the SAME bedtime story in the same tone and setting. "
                        f"Add new paragraphs to reach at least {TARGET_MIN_WORDS} words total. "
                        "Do NOT repeat earlier lines. Output ONLY the continuation content "
                        "without starting with <speak> or ending with </speak>. "
                        "Keep <break time=\"1.2s\"/> between paragraphs and occasional "
                        "<break time=\"400ms\"/> between sentences."
                    )
                    local_history.append({"role": "user", "content": cont_prompt})
                    cont = (run_model(model_key, local_history) or "").strip()
                    cont_inner = extract_inner_ssml(cont)

                    # Stitch with a paragraph break
                    inner = inner.rstrip() + '\n<break time="1.2s"/>\n' + cont_inner.lstrip()
                    wc = word_count(strip_ssml_tags(inner))
                    # Replace last assistant in local history with the updated stitched version
                    local_history[-2] = {"role": "assistant", "content": wrap_speak(inner)}
                    continues_left -= 1

                reply = normalize_speak_once(inner)
                logging.info(f"final_word_count={wc}")

            # Save final result (SSML stitched or original)
            save_assistant_message(convo_id, reply)
            mark_job_done(jid, failed=False, result_text=reply)
            logging.info(f"✅ Finished job {jid}")

        except Exception as e:
            logging.exception(f"❌ Failed job {jid if 'jid' in locals() else '?'}: {e}")
            try:
                if 'jid' in locals():
                    mark_job_done(jid, failed=True, result_text=str(e))
            except Exception:
                logging.exception("failed to mark job as error")
            time.sleep(0.5)  # small backoff on exceptions

def main():
    procs = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=worker_loop, args=(i + 1,))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == "__main__":
    main()
