"""Run only the pool tasks of armB.py (all 20 samples), cached per part."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import armB
from multiprocessing import Pool
if __name__ == "__main__":
    tasks = [t for t in armB.all_tasks() if t[0] == "pool"]
    print(len(tasks), "pool tasks", flush=True)
    with Pool(int(os.environ.get("NPROC", "8"))) as p:
        for name, status in p.imap_unordered(armB.run_task, tasks):
            print(time.strftime("%H:%M:%S"), name, status, flush=True)
