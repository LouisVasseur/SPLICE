#!/usr/bin/env python3
"""Run randomized, paired benchmark experiments. Python standard library only."""
from __future__ import annotations
import argparse, datetime, hashlib, itertools, json, os, pathlib, platform, random, subprocess, sys

def command_output(command: list[str]) -> str:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def environment(binary: pathlib.Path) -> dict:
    return {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "platform": platform.platform(), "machine": platform.machine(), "python": sys.version,
            "cpu": command_output(["lscpu"]), "compiler": command_output(["c++", "--version"]),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "binary_sha256": sha256(binary), "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "warning": "Shared-host exploratory results; CPU governor, NUMA binding and isolation not controlled by this script."}

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("build/scaleli_bench"))
    p.add_argument("--config", type=pathlib.Path, default=pathlib.Path("configs/smoke.json"))
    p.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/local"))
    p.add_argument("--repeats", type=int)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--cpu", type=int, help="Pin benchmark child to this Linux CPU using taskset.")
    p.add_argument("--resume", action="store_true", help="Append to an existing results.jsonl, skipping (dataset, profile, variant, seed, repeat) jobs already recorded.")
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()); binary = args.binary.resolve()
    if not binary.is_file(): p.error(f"missing binary: {binary}")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.jsonl"
    done = set()
    if result_path.exists():
        if not args.resume: p.error(f"refusing to overwrite {result_path}; choose a new output directory or pass --resume")
        for line in result_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["dataset"], r["profile"], r["variant"], r["seed"], r.get("repeat", 0)))
    repeats = args.repeats if args.repeats is not None else cfg.get("repeats", 1)
    if repeats < 1: p.error("repeats must be positive")
    env = environment(binary); env["config"] = cfg; env["requested_cpu"] = args.cpu
    if not (args.resume and (args.output / "environment.json").exists()): (args.output / "environment.json").write_text(json.dumps(env, indent=2))
    jobs = list(itertools.product(cfg["datasets"], cfg["profiles"], cfg["variants"], cfg.get("seeds", [42]), range(repeats)))
    random.Random(cfg.get("schedule_seed", 137)).shuffle(jobs)
    if done: print(f"resuming: {len(done)} of {len(jobs)} jobs already recorded", file=sys.stderr)
    with result_path.open("a" if args.resume else "w") as out:
        for j, (dataset, profile, variant, seed, repeat) in enumerate(jobs):
            dataset = {"name": dataset, "distribution": dataset} if isinstance(dataset, str) else dataset
            if (dataset["name"], profile, variant["name"], seed, repeat) in done: continue
            options = dict(cfg.get("common", {})); options.update({"profile": profile, "seed": seed})
            options.update({k: v for k, v in dataset.items() if k not in ("name", "flow_weights", "flow_train_args")})
            options.update({k: v for k, v in variant.items() if k != "name"})
            # A variant may say "flow": "$flow" to use the dataset's trained weights (see tools/prepare_flows.py).
            if options.get("flow") == "$flow":
                if "flow_weights" not in dataset: raise RuntimeError(f"variant {variant['name']} needs flow_weights on dataset {dataset['name']}")
                options["flow"] = str(pathlib.Path(dataset["flow_weights"]).resolve())
                if not pathlib.Path(options["flow"]).is_file(): raise RuntimeError(f"missing flow weights {options['flow']}; run tools/prepare_flows.py --config {args.config}")
            command = [str(binary)]
            for k, v in options.items(): command.extend(["--" + k, str(int(v)) if isinstance(v, bool) else str(v)])
            if args.cpu is not None: command = ["taskset", "-c", str(args.cpu)] + command
            proc = subprocess.run(command, text=True, capture_output=True, timeout=args.timeout)
            if proc.returncode:
                (args.output / "failed_command.json").write_text(json.dumps({"command": command, "stderr": proc.stderr}, indent=2))
                raise RuntimeError(f"benchmark failed: {proc.stderr}")
            row = json.loads(proc.stdout)
            row.update(dataset=dataset["name"], variant=variant["name"], repeat=repeat, command=command)
            out.write(json.dumps(row) + "\n"); out.flush()
            print(f"[{j+1}/{len(jobs)}] {dataset['name']} / {profile} / {variant['name']} seed={seed} repeat={repeat}", file=sys.stderr)
    print(str(result_path))
if __name__ == "__main__": main()
