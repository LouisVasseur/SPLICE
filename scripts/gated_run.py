#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Whole-node GRE runs: builds share the node, timed lookups never share it with anything but each other.

A batch starts as many runs as the memory of the four NUMA domains holds, each pinned to its own core. Each builds its
index and parks at the gate (gate.c), the last output before GRE's clock starts. Once every run is parked, nothing on the
node is working. They are then released one per domain, each moved first to its domain's timing core, so four lookups
run at a time on the four cores of the 4-lane baseline and the next batch waits for the last one to finish.

Pages stay on the domain that built them; the share that did is recorded per run. Build times from this script are
contended: only the throughput is comparable. w_* cells park before their warm-up, so it runs on the timing core.

  gated_run.py OUT --datasets "fb covid" --cells "splice lipp"   # resumable: runs already in OUT/runs.tsv are skipped
"""
import argparse, collections, csv, ctypes, glob, os, random, re, shutil, signal, subprocess, sys, time

# Jed: 72 cores, 4 NUMA domains, about 129 GB each
DOMAINS = [range(0, 18), range(18, 36), range(36, 54), range(54, 72)]
LANES = [2, 20, 38, 56]  # timing cores, one per domain
RESERVED = {0, 1}  # this script and whatever else lives in the job
FIELDS = "dataset index repeat throughput_ops memory_bytes success_read wall_s log lane batch build_s parked_s hwm_gb local_pct huge_gb".split()

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--datasets", required=True)
ap.add_argument("--cells", required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--ops", type=int, default=100000000)
ap.add_argument("--table", type=int, default=-1)
ap.add_argument("--budget-gb", type=float, default=105, help="memory of one domain that runs may take")
ap.add_argument("--peak-gb", type=float, help="first guess of a run's peak memory, for cells with no baseline")
ap.add_argument("--thp", action="store_true", help="keep huge pages (default: 4 KiB, as the baseline)")
ap.add_argument("--bin", default=os.environ.get("GRE_DIR", "") + "/build/microbench")
ap.add_argument("--data", default=os.environ.get("DATA_DIR", ""))
ap.add_argument("--baseline", default="", help="glob of runs.tsv files: expected build time, timed time and index size")
ap.add_argument("--gate-so", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "gate.so"))
a = ap.parse_args()

os.makedirs(a.out + "/logs", exist_ok=True)
gate_dir = f"/tmp/gate.{os.getpid()}"
os.makedirs(gate_dir)
libc = ctypes.CDLL(None, use_errno=True)
os.sched_setaffinity(0, {0})
random.seed(1866)
sha = subprocess.check_output(["sha256sum", a.bin], text=True)[:16]
run_bin = f"{a.out}/microbench.{sha}"  # the copy that runs, as in run_baselines.sh
if not os.path.exists(run_bin):
    shutil.copy2(a.bin, run_bin)
path = a.out + "/runs.tsv"
if not os.path.exists(path):
    open(path, "w").write("\t".join(FIELDS) + "\n")
done = {(r["dataset"], r["index"], int(r["repeat"])) for r in csv.DictReader(open(path), delimiter="\t")}

expected = {}
for f in glob.glob(a.baseline) if a.baseline else []:
    for r in csv.DictReader(open(f), delimiter="\t"):
        timed = a.ops / float(r["throughput_ops"])
        expected[r["dataset"], r["index"]] = (float(r["wall_s"]) - timed, timed, float(r["memory_bytes"]) / 1e9)
seen_peak, peak_ratio = {}, {}
placed = collections.defaultdict(lambda: [0, 0, 0, 0])


def first_guess(d, i):
    if a.peak_gb:
        return a.peak_gb
    return max(16, 10 + 1.6 * expected.get((d, i), (0, 0, 40))[2])


def peak_gb(d, i):  # measured, or the first guess scaled by what this index measured on another dataset
    if (d, i) in seen_peak:
        return seen_peak[d, i] * 1.02
    return first_guess(d, i) * peak_ratio[i] * 1.1 if i in peak_ratio else first_guess(d, i)


def plan(pending):  # longest builds first, so runs that take alike share a batch; repeats of a cell rotate over domains
    used, timed, batch = [0.0] * 4, [0.0] * 4, []
    free = [[c for c in cores if c not in RESERVED] for cores in DOMAINS]
    for d, i, r in sorted(pending, key=lambda x: -expected.get(x[:2], (1e9,))[0]):
        p, t = peak_gb(d, i), expected.get((d, i), (0, 30))[1]
        for k in sorted(range(4), key=lambda k: (placed[d, i][k], timed[k])):
            if used[k] + p <= a.budget_gb and free[k]:
                used[k] += p
                timed[k] += t
                placed[d, i][k] += 1
                batch.append(dict(d=d, i=i, r=r, k=k, core=free[k].pop()))
                break
    if not batch:
        sys.exit(f"nothing fits {a.budget_gb} GB per domain: {pending[:3]}")
    return batch


def start(x):
    tag = f"{x['d']}.{x['i']}.r{x['r']}"
    x["gate"], x["log"] = f"{gate_dir}/{tag}", f"logs/{tag}.log"

    def child():
        os.sched_setaffinity(0, {x["core"]})
        libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: no parked orphans
        if not a.thp and libc.prctl(41, 1, 0, 0, 0) != 0:  # PR_SET_THP_DISABLE
            os._exit(4)

    env = dict(os.environ, OMP_NUM_THREADS="1", LD_PRELOAD=a.gate_so, GATE=x["gate"])
    cmd = [run_bin, f"--keys_file={a.data}/{x['d']}", "--keys_file_type=binary", "--read=1", "--insert=0",
           f"--operations_num={a.ops}", f"--table_size={a.table}", "--init_table_ratio=1", "--thread_num=1", "--memory",
           f"--index={x['i']}", f"--output_path={a.out}/logs/{tag}.csv"]
    x["t0"] = time.time()
    x["proc"] = subprocess.Popen(cmd, stdout=open(f"{a.out}/{x['log']}", "w"), stderr=subprocess.STDOUT, env=env, preexec_fn=child)


def inspect(x):  # at the gate: peak memory, the share of pages on the run's own domain, huge pages
    pid, pages = x["proc"].pid, [0, 0, 0, 0]
    hwm = re.search(r"VmHWM:\s+(\d+)", open(f"/proc/{pid}/status").read())
    x["hwm_gb"] = int(hwm[1]) * 1024 / 1e9
    huge = re.search(r"AnonHugePages:\s+(\d+)", open(f"/proc/{pid}/smaps_rollup").read())
    x["huge_gb"] = int(huge[1]) * 1024 / 1e9
    for line in open(f"/proc/{pid}/numa_maps"):
        for n, c in re.findall(r" N(\d+)=(\d+)", line):
            pages[int(n)] += int(c)
    x["local_pct"] = 100 * pages[x["k"]] / max(1, sum(pages))
    seen_peak[x["d"], x["i"]] = x["hwm_gb"]
    peak_ratio[x["i"]] = max(peak_ratio.get(x["i"], 0), x["hwm_gb"] / first_guess(x["d"], x["i"]))


def release(x):
    pid = x["proc"].pid
    for tid in os.listdir(f"/proc/{pid}/task"):
        os.sched_setaffinity(int(tid), {LANES[x["k"]]})
    x["t_go"] = time.time()
    fd = os.open(x["gate"] + ".go", os.O_WRONLY)
    os.write(fd, b"x")
    os.close(fd)


def finish(x, batch_no):
    text = open(f"{a.out}/{x['log']}").read()
    get = lambda pat: (re.findall(pat, text) or [""])[-1]
    thr, mem, ok = get(r"Throughput = (\d+)"), get(r"Memory: (\d+)"), get(r"success_read: (\d+)")
    wall = time.time() - x["t0"]
    if x["proc"].returncode != 0 or ok != str(a.ops):
        print(f"FAILED {x['log']}: rc={x['proc'].returncode} success_read={ok}", flush=True)
        return
    row = [x["d"], x["i"], x["r"], thr, mem, ok, f"{wall:.0f}", x["log"], f"core{LANES[x['k']]}", batch_no, f"{x['t_ready'] - x['t0']:.1f}",
           f"{x['t_go'] - x['t_ready']:.1f}", f"{x['hwm_gb']:.2f}", f"{x['local_pct']:.2f}", f"{x['huge_gb']:.2f}"]
    with open(path, "a") as f:
        f.write("\t".join(map(str, row)) + "\n")


pending = [(d, i, r) for r in range(1, a.repeats + 1) for d in a.datasets.split() for i in a.cells.split() if (d, i, r) not in done]
t_all, batch_no = time.time(), 0
while pending:
    batch_no += 1
    batch = plan(pending)
    taken = {(x["d"], x["i"], x["r"]) for x in batch}
    pending = [p for p in pending if p not in taken]
    t0 = time.time()
    for x in batch:
        start(x)
    live = list(batch)
    while True:  # build: wait until every run is parked or dead
        waiting = 0
        for x in live:
            if "t_ready" in x:
                continue
            if os.path.exists(x["gate"] + ".ready"):
                x["t_ready"] = time.time()
            elif x["proc"].poll() is not None:
                print(f"FAILED before the gate {x['log']}: rc={x['proc'].returncode}", flush=True)
                x["dead"] = True
            else:
                waiting += 1
        live = [x for x in live if "dead" not in x]
        if not waiting:
            break
        time.sleep(0.5)
    t_built = time.time()
    time.sleep(1)  # let the last writer settle before anything is timed
    for x in live:
        inspect(x)
    queues = [[x for x in live if x["k"] == k] for k in range(4)]
    for q in queues:
        random.shuffle(q)
    running = [None] * 4
    while any(queues) or any(running):  # measure: one run per domain, nothing else working
        for k in range(4):
            if running[k] and running[k]["proc"].poll() is not None:
                finish(running[k], batch_no)
                running[k] = None
            if not running[k] and queues[k]:
                running[k] = queues[k].pop()
                release(running[k])
        time.sleep(0.2)
    t1 = time.time()
    worst = min((x["local_pct"] for x in live), default=0)
    print(f"batch {batch_no}: {len(batch)} runs, {sum(x['hwm_gb'] for x in live):.0f} GB, build {t_built - t0:.0f} s, measure {t1 - t_built:.0f} s, "
          f"worst local {worst:.1f}%, {len(pending)} runs left, {(t1 - t_all) / 60:.1f} min in all", flush=True)
shutil.rmtree(gate_dir, ignore_errors=True)
print(f"done in {(time.time() - t_all) / 60:.1f} min: {sha}", flush=True)
