#queue_worker.py
import multiprocessing
import time
import sys
from memory import (
    claim_next_job,
    get_conversation_messages,
    save_assistant_message,
    mark_job_done,
)
from model_runner import run_model

POLL_SEC = 1
NUM_WORKERS = 2

def worker_loop(worker_id: int):
    print(f"🚀 Worker {worker_id} started.")
    while True:
        job = claim_next_job()
        if not job:
            time.sleep(POLL_SEC)
            continue

        jid = job["id"]
        convo_id = job["conversation_id"]
        model_key = job["model"]
        user_input = job["user_input"]
        system_prompt = job.get("system_prompt") or ""

        try:
            history = get_conversation_messages(convo_id)
            if not history or history[-1]["role"] != "user" or history[-1]["content"] != user_input:
                history.append({"role": "user", "content": user_input})

            if system_prompt.strip():
                history.insert(0, {"role": "system", "content": system_prompt.strip()})

            reply = run_model(model_key, history)
            save_assistant_message(convo_id, reply)
            mark_job_done(jid, failed=False, result_text=reply)

            print(f"[Worker {worker_id}] ✅ Finished job {jid}")
        except Exception as e:
            mark_job_done(jid, failed=True, result_text=str(e))
            print(f"[Worker {worker_id}] ❌ Failed job {jid}: {e}", file=sys.stderr)

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
