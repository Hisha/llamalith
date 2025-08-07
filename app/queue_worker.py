import multiprocessing
import time
from app.db import get_next_job, mark_job_done, save_assistant_message
from app.model_runner import run_model

def worker_loop(worker_id):
    print(f"🚀 Worker {worker_id} started.")
    while True:
        job = get_next_job()
        if job:
            print(f"[Worker {worker_id}] Processing job {job['id']}...")
            try:
                assistant_output = run_model(
                    job['model'],
                    job['messages']
                )
                save_assistant_message(job['conversation_id'], assistant_output)
                mark_job_done(job['id'])
                print(f"[Worker {worker_id}] ✅ Finished job {job['id']}")
            except Exception as e:
                print(f"[Worker {worker_id}] ❌ Failed job {job['id']}: {e}")
                mark_job_done(job['id'], failed=True)
        else:
            time.sleep(1)

if __name__ == "__main__":
    num_workers = 2  # ← Adjust this to fit your CPU load
    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(target=worker_loop, args=(i + 1,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
