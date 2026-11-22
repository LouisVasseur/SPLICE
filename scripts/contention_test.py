#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Do the four timing lanes slow each other down? One batch of builds, then each cell on each lane twice: alone on the
node, then on all four lanes at once (the most it can contend with itself). Same parking and moves as gated_run.py.

  contention_test.py OUT.tsv covid:splice fb:spl_tv
"""
import ctypes, os, random, re, signal, subprocess, sys, time

LANES, BIN, DATA, SO = [2, 20, 38, 56], os.environ["GRE_DIR"] + "/build/microbench", os.environ["DATA_DIR"], os.path.dirname(os.path.abspath(__file__)) + "/gate.so"
out, cells = sys.argv[1], [c.split(":") for c in sys.argv[2:]]
gate_dir = f"/tmp/ctest.{os.getpid()}"
os.makedirs(gate_dir)
libc = ctypes.CDLL(None, use_errno=True)
os.sched_setaffinity(0, {0})
random.seed(1866)
runs, next_core = [], [17, 35, 53, 71]


def start(d, i, k, phase):
    tag = f"{d}.{i}.{phase}.{k}"
    x = dict(d=d, i=i, k=k, phase=phase, gate=f"{gate_dir}/{tag}", log=f"{gate_dir}/{tag}.log")
    core = next_core[k]
    next_core[k] -= 1

    def child():
        os.sched_setaffinity(0, {core})
        libc.prctl(1, signal.SIGKILL)
        if libc.prctl(41, 1, 0, 0, 0) != 0:
            os._exit(4)

    env = dict(os.environ, OMP_NUM_THREADS="1", LD_PRELOAD=SO, GATE=x["gate"])
    cmd = [BIN, f"--keys_file={DATA}/{d}", "--keys_file_type=binary", "--read=1", "--insert=0", "--operations_num=100000000", "--table_size=-1",
           "--init_table_ratio=1", "--thread_num=1", "--memory", f"--index={i}", f"--output_path={x['log']}.csv"]
    x["proc"] = subprocess.Popen(cmd, stdout=open(x["log"], "w"), stderr=subprocess.STDOUT, env=env, preexec_fn=child)
    return x


def release(x):
    for tid in os.listdir(f"/proc/{x['proc'].pid}/task"):
        os.sched_setaffinity(int(tid), {LANES[x["k"]]})
    fd = os.open(x["gate"] + ".go", os.O_WRONLY)
    os.write(fd, b"x")
    os.close(fd)


def finish(x):
    x["proc"].wait()
    thr = re.findall(r"Throughput = (\d+)", open(x["log"]).read())[-1]
    with open(out, "a") as f:
        f.write(f"{x['phase']}\t{x['d']}\t{x['i']}\tcore{LANES[x['k']]}\t{thr}\n")
    print(x["phase"], x["d"], x["i"], LANES[x["k"]], thr, flush=True)


for d, i in cells:
    for phase in ("solo", "all4"):
        for k in range(4):
            runs.append(start(d, i, k, phase))
t0 = time.time()
while not all(os.path.exists(x["gate"] + ".ready") for x in runs):
    if any(x["proc"].poll() is not None for x in runs):
        sys.exit("a run died before the gate")
    time.sleep(0.5)
print(f"built in {time.time() - t0:.0f} s", flush=True)
time.sleep(1)
solo = [x for x in runs if x["phase"] == "solo"]
random.shuffle(solo)
for x in solo:
    release(x)
    finish(x)
for d, i in cells:
    group = [x for x in runs if x["phase"] == "all4" and (x["d"], x["i"]) == (d, i)]
    for x in group:
        release(x)
    for x in group:
        finish(x)
print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)
