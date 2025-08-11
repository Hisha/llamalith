# queue_worker.py
import multiprocessing
import time
import sys
import logging, os, traceback

from memory import (
    claim_next_job,
    get_conversation_messages,
    save_assistant_message,
    mark_job_done,
)
from model_runner import run_model

POLL_SEC = 1
NUM_WORKERS = 2

logging.basicConfig(
    level=logging.INFO,
    format='[Worker %(process)d] %(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

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

            logging.info(f"claimed job={jid} model={model_key} convo={convo_id} ulen={len(user_input)}")

            # Build history
            history = get_conversation_messages(convo_id) or []

            # Ensure current user turn is present for this run
            if user_input and (not history or history[-1].get("role") != "user" or history[-1].get("content","").strip() != user_input):
                history.append({"role": "user", "content": user_input})

            # Ensure system prompt at the top (avoid dup)
            if system_prompt and (not history or history[0].get("role") != "system" or history[0].get("content","").strip() != system_prompt):
                history.insert(0, {"role": "system", "content": system_prompt})

            logging.info(f"history_turns={len(history)}; calling model…")
            reply = run_model(model_key, history)
            reply = (reply or "").strip()
            logging.info(f"reply_len={len(reply)}")

            # Guardrail: don't save blanks silently
            if not reply:
                mark_job_done(jid, failed=True, result_text="Empty model output")
                logging.warning(f"empty output -> marked job {jid} failed")
                continue

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
