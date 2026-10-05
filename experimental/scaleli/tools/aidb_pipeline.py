#!/usr/bin/env python3
"""AIDB 2026 hardness-protocol lane: idempotent end-to-end pipeline. Python standard library only.

Applies the protocol of Zhang, Tang and Ailamaki, "How Hard Can Indexing Be? Principled Dataset
Hardness Measurement for Learned Indexes" (AIDB @ VLDB 2026) to THIS repository's clean-room
controls (NFL-style flow transform, CSV-style virtual points) on the ten GRE datasets, at a
documented REDUCED scale:

  * hardness metrics (RMSE, ME, CD, PLA-32, PLA-4096) on the full 200M keys  -> scaleli_hardness (C1)
  * index runs on 2M-key uniform samples (seed 42) of every dataset          -> scaleli_bench via tools/run_suite.py
  * conformance / coverage of the 25 Table-2 metrics against OUR variants    -> tools/aidb_scores.py (C3)
  * one self-contained HTML report with inline SVG                          -> tools/aidb_report.py (C4)

Every number produced here is a single-host, reduced-scale, clean-room CONTROL measurement.
It is NOT a reproduction of the paper, of NFL/AFLI, or of CSV.

Steps run in this order and each one checks its outputs first (--force redoes them):
  verify-downloads  sort  sample  flows  hardness  sweep  throughput  scores  report
The pipeline never downloads anything. Datasets are expected under --data-dir as SOSD files
named <name>; a <name>.part file means a download is still in progress (--wait polls for it).
Seven of the ten GRE files are served UNSORTED (covid, genome, history, libio, planet, stack,
wise; GRE's own loader sorts and de-duplicates at load time), so the `sort` step audits every
file with `scaleli_hardness --sort-only` and writes a sorted, de-duplicated copy <name>.sorted
(1.6 GB each, 11.2 GB for the seven) that every later step uses; provenance.json keeps the
download's own sha256 and, under sort_audit, the copy's sha256.
The sweep pairs the eight contracted variants (sorted_vector raw_rank packed_rank packed_rank_flow
packed_rank_flow_forced packed_rank_vp10 packed_rank_flow_vp10 packed_byte); `--extra-variants fusion`
adds the two cost-based fusion controls, which need a scaleli_bench that accepts --fusion.
--dry-run exercises every step on the bundled fixture plus four synthetic key sets (one of them
shuffled with duplicates, to exercise the sort step) in under two minutes, writing to
results/aidb_dry/, and needs no network.
"""
from __future__ import annotations
import argparse, array, collections, concurrent.futures, datetime, hashlib, json, pathlib, random, shlex, shutil, statistics, struct, subprocess, sys, threading, time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                    # experimental/scaleli
SOTA = ROOT.parents[1]                # repository root
FIXTURE = SOTA / "evidence/dense_sparse_fixture_uint64"
GRE_URL = "https://www.cse.cuhk.edu.hk/mlsys/gre/"
GRE_DATASETS = ["books", "fb", "osm", "covid", "genome", "history", "libio", "planet", "stack", "wise"]
GRE_UNSORTED = ["covid", "genome", "history", "libio", "planet", "stack", "wise"]   # served unsorted on disk (audited 2026-09-21)
# Dry-run datasets: the bundled fixture, three synthetic sets, and one synthetic set that is shuffled
# and carries duplicates so that the sort step, the sorted-copy provenance and the stale-sample check run.
DRY_DATASETS = {"dense_sparse_fixture": {"fixture": True},
                "lognormal": {"distribution": "lognormal", "seed": 42},
                "clustered": {"distribution": "clustered", "seed": 42},
                "locally_hard": {"distribution": "locally_hard", "seed": 42},
                "shuffled_dups": {"distribution": "lognormal", "seed": 7, "shuffle": True, "duplicates": 500}}
DRY_N = 50000
STEPS = ["verify-downloads", "sort", "sample", "flows", "hardness", "sweep", "throughput", "scores", "report"]
VARIANTS = [   # contract C5: exactly these eight
    {"name": "sorted_vector", "index": "sorted_vector"},
    {"name": "raw_rank", "policy": "raw", "routing": "rank"},
    {"name": "packed_rank", "policy": "min_bytes", "routing": "rank"},
    {"name": "packed_rank_flow", "policy": "min_bytes", "routing": "rank", "flow": "$flow", "flow-bypass": 1},
    {"name": "packed_rank_flow_forced", "policy": "min_bytes", "routing": "rank", "flow": "$flow", "flow-bypass": 0},
    {"name": "packed_rank_vp10", "policy": "min_bytes", "routing": "rank", "virtual-alpha": 0.1},
    {"name": "packed_rank_flow_vp10", "policy": "min_bytes", "routing": "rank", "flow": "$flow", "flow-bypass": 1, "virtual-alpha": 0.1},
    {"name": "packed_byte", "policy": "min_bytes", "routing": "byte"},
]
EXTRA_VARIANTS = {   # opt-in (--extra-variants); each group names the scaleli_bench option it needs
    "fusion": {"requires": "--fusion", "variants": [
        {"name": "packed_rank_fusion_auto", "policy": "min_bytes", "routing": "rank", "flow": "$flow", "virtual-alpha": 0.1, "fusion": "auto"},
        {"name": "packed_rank_flow_costsel", "policy": "min_bytes", "routing": "rank", "flow": "$flow", "fusion": "auto"}]},
}
LABEL = ("Reduced-scale, single-host, clean-room controls of this repository's NFL-style transform and CSV-style "
         "virtual points under the AIDB 2026 hardness protocol. Not a reproduction of the paper, NFL/AFLI or CSV.")


class StepError(RuntimeError):
    def __init__(self, step: str, message: str):
        super().__init__(message); self.step = step


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def sosd_count(path: pathlib.Path) -> int:
    with path.open("rb") as f: b = f.read(8)
    if len(b) != 8: raise ValueError(f"{path}: truncated SOSD count header")
    return struct.unpack("<Q", b)[0]


def sosd_keys(path: pathlib.Path) -> array.array:
    """All uint64 keys of a (small) SOSD file as an array; meant for samples and dry-run sets, not 200M-key files."""
    n = sosd_count(path); keys = array.array("Q")
    with path.open("rb") as f: f.seek(8); keys.frombytes(f.read(8 * n))
    if sys.byteorder != "little": keys.byteswap()
    if len(keys) != n: raise ValueError(f"{path}: header says {n:,} keys, file holds {len(keys):,}")
    return keys


def sosd_strictly_ascending(path: pathlib.Path) -> bool:
    keys = sosd_keys(path)
    return all(a < b for a, b in zip(keys, keys[1:]))


def write_sosd(path: pathlib.Path, keys: array.array) -> None:
    out = array.array("Q", keys)
    if sys.byteorder != "little": out.byteswap()
    with path.open("wb") as f: f.write(struct.pack("<Q", len(out))); f.write(out.tobytes())


def size_label(n: int) -> str:
    if n % 1_000_000 == 0: return f"{n // 1_000_000}M"
    if n % 1_000 == 0: return f"{n // 1_000}K"
    return str(n)


def fresh(output: pathlib.Path, *inputs: pathlib.Path) -> bool:
    """True when output exists and is at least as new as every existing input."""
    if not output.is_file(): return False
    return all(not p.is_file() or p.stat().st_mtime_ns <= output.stat().st_mtime_ns for p in inputs)


def read_json(path: pathlib.Path) -> dict:
    try: return json.loads(path.read_text())
    except (OSError, ValueError) as e: raise StepError("io", f"cannot read {path}: {e}")


def write_json(path: pathlib.Path, obj) -> bool:
    """Atomic write; returns False and leaves the file (and its mtime) alone when the content is unchanged."""
    text = json.dumps(obj, indent=2, sort_keys=False)
    try:
        if path.is_file() and path.read_text() == text: return False
    except OSError: pass
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); tmp.write_text(text); tmp.replace(path); return True


def fmt(x, digits: int = 2) -> str:
    return f"{x:.{digits}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else "n/a"


def metric_block(block: dict | None, extra: tuple = ()) -> dict | None:
    """Contract C3 hardness scope block from a contract C1 metric block, plus the listed degeneracy fields
    (keys, duplicates, unordered_pairs, virtual_points, regions) when the raw block has them; readers ignore extras."""
    if not isinstance(block, dict): return None
    pla = block.get("pla") or {}
    try:
        out = {"rmse": float(block["rmse"]), "max_error": float(block["max_error"]), "conflict_degree": float(block["conflict_degree"]),
               "pla_32": float(pla.get("32", pla.get(32))), "pla_4096": float(pla.get("4096", pla.get(4096)))}
    except (KeyError, TypeError, ValueError) as e:
        raise StepError("hardness", f"metric block lacks a contract C1 field ({e}): {json.dumps(block)[:300]}")
    for k in ("keys",) + tuple(extra):
        if k in block: out[k] = block[k]
    return out


def median_throughput(rows: list[dict]) -> tuple[dict, dict]:
    """Contract C3 T.json ({variant: {dataset: median MOPS}}) plus a per-seed detail map; checks trace pairing."""
    fingerprints: dict[tuple, set] = collections.defaultdict(set)
    values: dict[tuple, list] = collections.defaultdict(list)
    for r in rows:
        fingerprints[(r["dataset"], r["profile"], r["seed"], r["repeat"])].add(r["trace_fingerprint"])
        values[(r["variant"], r["dataset"])].append({"seed": r["seed"], "repeat": r["repeat"], "mops": r["throughput_ops_s"] / 1e6,
                                                     "verified": r.get("verified"), "operations": r.get("operations")})
    bad = [k for k, s in fingerprints.items() if len(s) != 1]
    if bad: raise StepError("throughput", f"unpaired traces across variants for {bad[:3]}")
    order = [v["name"] for v in VARIANTS] + [v["name"] for g in EXTRA_VARIANTS.values() for v in g["variants"]]
    variants = sorted({v for v, _ in values}, key=lambda v: (order.index(v) if v in order else len(order), v))
    T = {v: {d: statistics.median(x["mops"] for x in values[(v, d)]) for (vv, d) in sorted(values) if vv == v} for v in variants}
    detail = {v: {d: values[(v, d)] for (vv, d) in sorted(values) if vv == v} for v in variants}
    return T, detail


class Pipeline:
    def __init__(self, a: argparse.Namespace):
        self.a = a; self.dry = a.dry_run
        self.results = a.results.resolve(); self.results.mkdir(parents=True, exist_ok=True)
        self.log_path = self.results / "pipeline.log"; self.lock = threading.Lock()
        self.data_dir = a.data_dir.resolve(); self.samples_dir = a.samples_dir.resolve()
        self.binary_dir = a.binary_dir.resolve(); self.bench = self.binary_dir / "scaleli_bench"; self.hardness_bin = self.binary_dir / "scaleli_hardness"
        self.datasets = list(a.datasets)
        self.catalog = {e["id"]: e for e in read_json(ROOT / "configs/datasets.json")["datasets"]}
        self.provenance_path = self.results / "provenance.json"
        self.config_path = (self.results / "aidb.json") if self.dry else (ROOT / "configs/aidb.json")
        self.sweep_dir = self.results / "sweep"
        self.variants = list(VARIANTS) + [v for g in a.extra_variants for v in EXTRA_VARIANTS[g]["variants"]]
        self._hashes: dict[str, str] = {}

    # ----- utilities -------------------------------------------------------------------------
    def log(self, step: str, message: str) -> None:
        line = f"{utc_now()} [{step}] {message}"
        with self.lock:
            with self.log_path.open("a") as f: f.write(line + "\n")
            print(line, file=sys.stderr, flush=True)

    def file_hash(self, path) -> str:
        p = pathlib.Path(path).resolve(); key = str(p)
        if key not in self._hashes: self._hashes[key] = sha256_file(p)
        return self._hashes[key]

    def fingerprint(self, cmd: list[str]) -> dict:
        """sha256 of the executable and script behind a command (the first two arguments that are files)."""
        return {c: self.file_hash(c) for c in cmd[:2] if pathlib.Path(c).is_file()}

    def run(self, step: str, cmd: list, cwd: pathlib.Path = ROOT, timeout: float | None = None) -> subprocess.CompletedProcess:
        cmd = [str(x) for x in cmd]; self.log(step, "run: " + " ".join(shlex.quote(c) for c in cmd)); t0 = time.perf_counter()
        try: p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout or self.a.timeout)
        except subprocess.TimeoutExpired: raise StepError(step, f"timed out after {timeout or self.a.timeout:.0f} s: {cmd[0]}")
        except OSError as e: raise StepError(step, f"cannot start {cmd[0]}: {e}")
        dt = time.perf_counter() - t0
        if p.returncode:
            tail = (p.stderr or p.stdout or "").strip()[-2000:]
            self.log(step, f"FAILED exit={p.returncode} after {dt:.1f} s: {pathlib.Path(cmd[0]).name}\n{tail}")
            raise StepError(step, f"{pathlib.Path(cmd[0]).name} exited {p.returncode}; see {self.log_path}")
        self.log(step, f"ok {dt:.1f} s: {pathlib.Path(cmd[0]).name}"); return p

    def short(self, path: pathlib.Path) -> str:
        try: return str(path.resolve().relative_to(self.results))
        except ValueError: return str(path)

    @staticmethod
    def identity_key(identity: dict) -> list:
        """The command with its fingerprinted executable/script replaced by their sha256: a byte-identical binary at
        another path is the same identity, the same path holding a rebuilt binary is a different one."""
        fp = identity.get("sha256") or {}
        return [fp.get(c, c) for c in identity.get("command") or []]

    def cached_run(self, step: str, output: pathlib.Path, cmd: list, timeout: float | None = None, force: bool = False) -> bool:
        """Run cmd writing its stdout to output unless output exists from the identical command AND the identical
        executable/script (sha256 in the .command.json side file). Returns True when run."""
        cmd = [str(x) for x in cmd]; side = output.with_name(output.name + ".command.json")
        identity = {"command": cmd, "sha256": self.fingerprint(cmd)}
        if output.is_file() and side.is_file() and not self.a.force and not force:
            try: saved = json.loads(side.read_text()); ok = bool(json.loads(output.read_text()))
            except (OSError, ValueError): saved, ok = None, False
            if ok and isinstance(saved, dict) and self.identity_key(saved) == self.identity_key(identity):
                self.log(step, f"keep {self.short(output)}"); return False
            if ok and isinstance(saved, list) and saved == cmd:
                self.log(step, f"keep {self.short(output)} (WARNING: predates binary fingerprinting; --force to redo)"); return False
            if ok and isinstance(saved, dict) and (saved.get("command") or [])[1:] == cmd[1:]:
                self.log(step, f"redo {self.short(output)}: same arguments, different binary/script sha256 ({', '.join(f'{pathlib.Path(k).name} {v[:12]}…' for k, v in identity['sha256'].items())})")
        output.parent.mkdir(parents=True, exist_ok=True); p = self.run(step, cmd, timeout=timeout)
        try: json.loads(p.stdout)
        except ValueError: raise StepError(step, f"{cmd[0]} did not print a JSON object (see {self.log_path})")
        output.write_text(p.stdout); side.write_text(json.dumps(identity, indent=2)); return True

    def need_binary(self, step: str, path: pathlib.Path, target: str) -> None:
        if not path.is_file(): raise StepError(step, f"missing {path}; build the CMake target {target} (see --binary-dir)")

    def bench_accepts(self, option: str) -> bool:
        try: p = subprocess.run([str(self.bench), "--help"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired): return False
        return option in (p.stdout + p.stderr)

    def rel(self, path: pathlib.Path) -> str:
        try: return str(path.resolve().relative_to(ROOT))
        except ValueError: return str(path.resolve())

    def load_provenance(self) -> dict:
        return read_json(self.provenance_path) if self.provenance_path.is_file() else {}

    def save_provenance(self, prov: dict) -> None:
        write_json(self.provenance_path, prov)

    # ----- dataset bookkeeping --------------------------------------------------------------
    def catalog_entry(self, name: str) -> dict:
        return {} if self.dry else self.catalog.get("gre_" + name, {})

    def raw_path(self, name: str) -> pathlib.Path:
        """The download itself (provenance: url, sha256, bytes, mtime are always those of this file)."""
        return self.data_dir / name

    def sorted_path(self, name: str) -> pathlib.Path:
        return self.data_dir / (name + ".sorted")

    def dataset_path(self, name: str) -> pathlib.Path:
        """The file every step after verify-downloads works on: the sorted, de-duplicated copy written by the sort
        step when the download is unsorted or has duplicates (GRE serves covid, genome, history, libio, planet,
        stack and wise unsorted), else the download itself."""
        sorted_copy = self.sorted_path(name)
        return sorted_copy if sorted_copy.is_file() else self.raw_path(name)

    def expected_count(self, name: str) -> int | None:
        return self.catalog_entry(name).get("count")

    def url(self, name: str) -> str:
        if not self.dry: return self.catalog_entry(name).get("url", GRE_URL + name)
        spec = DRY_DATASETS.get(name, {})
        if spec.get("fixture"): return f"file://{FIXTURE}"
        extra = "&shuffled=1" if spec.get("shuffle") else ""
        extra += f"&duplicates={spec['duplicates']}" if spec.get("duplicates") else ""
        return f"synthetic://scaleli_bench?distribution={spec.get('distribution')}&n={DRY_N}&seed={spec.get('seed', 42)}{extra}"

    def complete(self, name: str) -> bool:
        path = self.raw_path(name)
        if not path.is_file() or path.with_name(name + ".part").exists(): return False
        try: return path.stat().st_size == 8 + 8 * sosd_count(path)
        except (OSError, ValueError): return False

    def progress(self, name: str) -> str:
        part = self.raw_path(name).with_name(name + ".part"); expected = self.expected_count(name)
        if part.is_file():
            got = part.stat().st_size
            return f"{got:,} bytes" if not expected else f"{100 * got / (8 + 8 * expected):.1f}%"
        return "present, size mismatch" if self.raw_path(name).is_file() else "missing"

    def sample_n(self, name: str) -> int:
        return min(self.a.sample, sosd_count(self.dataset_path(name)))

    def sample_path(self, name: str) -> pathlib.Path:
        return self.samples_dir / f"{name}_{size_label(self.sample_n(name))}_{self.a.sample_mode}_s{self.a.sample_seed}"

    def flow_path(self, name: str) -> pathlib.Path:
        return self.results / "flows" / f"{name}_2D2H2L.txt"

    def source_sha256(self, name: str, src: pathlib.Path, entry: dict) -> str | None:
        """Known sha256 of the file a sample must have been drawn from, or None when provenance does not have it yet."""
        if src == self.sorted_path(name): return (entry.get("sort_audit") or {}).get("sha256")
        return entry.get("sha256")

    # ----- steps ---------------------------------------------------------------------------
    def materialize_dry_datasets(self, step: str) -> None:
        self.need_binary(step, self.bench, "scaleli_bench"); self.data_dir.mkdir(parents=True, exist_ok=True)
        for name in self.datasets:
            dest = self.raw_path(name); spec = DRY_DATASETS[name]
            if dest.is_file() and not self.a.force: continue
            if self.a.force and self.sorted_path(name).exists(): self.sorted_path(name).unlink()
            if spec.get("fixture"):
                if not FIXTURE.is_file(): raise StepError(step, f"missing fixture {FIXTURE}")
                shutil.copyfile(FIXTURE, dest); self.log(step, f"{name}: copied fixture {FIXTURE.name}")
                continue
            tmp = dest.with_name(name + ".part")
            self.run(step, [self.bench, "--distribution", spec["distribution"], "--n", DRY_N, "--seed", spec.get("seed", 42), "--ops", 1, "--profile", "read_only", "--load-ratio", 1,
                            "--verify", 0, "--instrument", 0, "--warmup", 0, "--dump-keys", tmp])
            note = ""
            if spec.get("shuffle") or spec.get("duplicates"):   # an unsorted file with duplicates, as GRE serves some of its files
                keys = sosd_keys(tmp); rng = random.Random(spec.get("seed", 42))
                dups = [keys[rng.randrange(len(keys))] for _ in range(int(spec.get("duplicates", 0)))]
                keys = array.array("Q", list(keys) + dups)
                if spec.get("shuffle"): rng.shuffle(keys)
                write_sosd(tmp, keys); note = f", then shuffled with {len(dups)} duplicates appended ({len(keys):,} keys on disk)"
            tmp.replace(dest); self.log(step, f"{name}: generated {DRY_N:,} synthetic {spec['distribution']} keys{note}")

    def step_verify_downloads(self) -> None:
        step = "verify-downloads"
        if self.dry: self.materialize_dry_datasets(step)
        pending = [d for d in self.datasets if not self.complete(d)]
        while pending:
            status = ", ".join(f"{d} ({self.progress(d)})" for d in pending)
            if not self.a.wait: raise StepError(step, f"incomplete datasets under {self.data_dir}: {status}. Finish the download or pass --wait; this pipeline never downloads.")
            self.log(step, f"waiting {self.a.poll_seconds} s for {status}"); time.sleep(self.a.poll_seconds)
            pending = [d for d in self.datasets if not self.complete(d)]
        prov = self.load_provenance()
        for d in self.datasets:
            path = self.raw_path(d); st = path.stat(); n = sosd_count(path); entry = prov.get(d, {})   # always the download, never the sorted copy
            if not self.a.force and entry.get("sha256") and entry.get("bytes") == st.st_size and entry.get("mtime_ns") == st.st_mtime_ns and entry.get("path") == str(path):
                self.log(step, f"{d}: keep provenance ({n:,} keys, sha256 {entry['sha256'][:16]}…)"); continue
            expected = self.expected_count(d)
            if expected is not None and n != expected: raise StepError(step, f"{d}: header count {n:,} differs from catalog count {expected:,}")
            t0 = time.perf_counter(); digest = sha256_file(path)
            entry.update({"url": self.url(d), "source": self.catalog_entry(d).get("source", "local"), "path": str(path), "format": "sosd", "dtype": "uint64",
                          "count": n, "bytes": st.st_size, "sha256": digest, "mtime_ns": st.st_mtime_ns,
                          "retrieved_utc": datetime.datetime.fromtimestamp(st.st_mtime, datetime.timezone.utc).isoformat(timespec="seconds"),
                          "verified_utc": utc_now(), "role": self.catalog_entry(d).get("role", "synthetic or bundled fixture (dry run)")})
            prov[d] = entry; self.save_provenance(prov)
            self.log(step, f"{d}: {n:,} keys, {st.st_size:,} bytes, sha256 {digest[:16]}… ({time.perf_counter() - t0:.1f} s)")

    def sorted_copy_record(self, path: pathlib.Path) -> dict:
        st = path.stat(); t0 = time.perf_counter(); digest = sha256_file(path)
        return {"sha256": digest, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "hash_seconds": round(time.perf_counter() - t0, 1)}

    def step_sort(self) -> None:
        """Audit sortedness/duplicates with scaleli_hardness --sort-only; write and checksum <name>.sorted for unsorted or duplicate files."""
        step = "sort"; self.need_binary(step, self.hardness_bin, "scaleli_hardness"); prov = self.load_provenance()
        for d in self.datasets:
            raw = self.raw_path(d); sorted_copy = self.sorted_path(d); entry = prov.setdefault(d, {}); audit = entry.get("sort_audit") or {}
            if not raw.is_file(): raise StepError(step, f"{d}: missing {raw}; run verify-downloads first")
            if audit and not self.a.force and audit.get("raw_sha256", entry.get("sha256")) == entry.get("sha256"):
                if audit.get("sorted") and not audit.get("written"):
                    self.log(step, f"{d}: keep audit (sorted on disk, duplicates={audit.get('duplicates')})"); continue
                if sorted_copy.is_file():
                    if not audit.get("sha256") or audit.get("bytes") != sorted_copy.stat().st_size or audit.get("mtime_ns") != sorted_copy.stat().st_mtime_ns:
                        audit.update(self.sorted_copy_record(sorted_copy)); entry["sort_audit"] = audit; self.save_provenance(prov)
                        self.log(step, f"{d}: checksummed existing {sorted_copy.name} (sha256 {audit['sha256'][:16]}…, {audit['hash_seconds']} s)")
                    self.log(step, f"{d}: keep audit (sorted={audit.get('sorted')}, duplicates={audit.get('duplicates')}, using {sorted_copy.name})"); continue
                self.log(step, f"{d}: audit says a sorted copy was written but {sorted_copy.name} is missing; redoing the sort")
            if sorted_copy.is_file() and not self.a.force:  # another results dir already produced it; never rewrite a copy in use
                entry["sort_audit"] = {"sorted": False, "duplicates": None, "written": str(sorted_copy), "written_keys": sosd_count(sorted_copy),
                                       "raw_sha256": entry.get("sha256"), "note": "sorted copy already present; audit reused", **self.sorted_copy_record(sorted_copy)}
                self.save_provenance(prov); self.log(step, f"{d}: reuse existing {sorted_copy.name} ({sosd_count(sorted_copy):,} keys, sha256 {entry['sort_audit']['sha256'][:16]}…)"); continue
            if sorted_copy.exists(): sorted_copy.unlink()
            audit_file = self.results / "sort" / f"{d}.json"
            self.cached_run(step, audit_file, [self.hardness_bin, "--data", raw, "--dtype", "uint64", "--check-sorted", 0, "--sort-only", 1, "--write-sorted", sorted_copy], force=True)
            r = read_json(audit_file)
            if r.get("written") and not sorted_copy.is_file(): raise StepError(step, f"{d}: scaleli_hardness reported {r['written']} but it is missing")
            entry["sort_audit"] = {"sorted": r.get("sorted"), "duplicates": r.get("duplicates"), "written": r.get("written"), "written_keys": r.get("written_keys"),
                                   "sort_ns": r.get("sort_ns"), "threads": r.get("threads"), "raw_sha256": entry.get("sha256")}
            if r.get("written"): entry["sort_audit"].update(self.sorted_copy_record(sorted_copy))
            self.save_provenance(prov)
            self.log(step, f"{d}: sorted={r.get('sorted')} duplicates={r.get('duplicates')}"
                     + (f" -> wrote {sorted_copy.name} ({r.get('written_keys'):,} keys, sort {float(r.get('sort_ns') or 0) / 1e9:.1f} s, sha256 {entry['sort_audit']['sha256'][:16]}…)" if r.get("written") else ""))
        self.save_provenance(prov)

    def sample_stale_reason(self, d: str, src: pathlib.Path, dest: pathlib.Path, manifest: pathlib.Path, entry: dict) -> str | None:
        """Why an existing sample must be redrawn, or None when it matches the current source file and settings."""
        if not dest.is_file() or not manifest.is_file(): return "missing"
        try: m = json.loads(manifest.read_text())
        except (OSError, ValueError) as e: return f"unreadable manifest ({e})"
        if m.get("source") != str(src): return f"drawn from {m.get('source')}, current source is {src}"
        expected = self.source_sha256(d, src, entry)
        if expected and m.get("source_sha256") != expected: return f"source sha256 {str(m.get('source_sha256'))[:16]}… differs from provenance {expected[:16]}…"
        if (m.get("sample_count"), m.get("mode"), m.get("seed")) != (self.sample_n(d), self.a.sample_mode, self.a.sample_seed): return "sample size, mode or seed changed"
        if not sosd_strictly_ascending(dest): return "sample keys are not strictly ascending"
        return None

    def step_sample(self) -> None:
        step = "sample"; prov = self.load_provenance(); self.samples_dir.mkdir(parents=True, exist_ok=True)
        for d in self.datasets:
            src = self.dataset_path(d); entry = prov.setdefault(d, {})
            if not src.is_file(): raise StepError(step, f"{d}: missing {src}; run verify-downloads first")
            dest = self.sample_path(d); manifest = dest.with_name(dest.name + ".manifest.json")
            why = "forced" if self.a.force else self.sample_stale_reason(d, src, dest, manifest, entry)
            if why is None: self.log(step, f"{d}: keep {dest.name} ({sosd_count(dest):,} keys)")
            else:
                if why != "missing": self.log(step, f"{d}: redo sample: {why}")
                for p in (dest, manifest):
                    if p.exists(): p.unlink()
                self.run(step, [sys.executable, HERE / "datasets.py", "sample", src, dest, "--n", self.sample_n(d), "--dtype", "uint64", "--mode", self.a.sample_mode, "--seed", self.a.sample_seed])
                self.log(step, f"{d}: wrote {dest.name} ({sosd_count(dest):,} keys from {src.name})")
            m = read_json(manifest); m["path"] = str(dest); entry["sample"] = m
        self.save_provenance(prov)

    def step_flows(self) -> None:
        step = "flows"; (self.results / "flows").mkdir(parents=True, exist_ok=True); reports = []
        for d in self.datasets:
            sample = self.sample_path(d)
            if not sample.is_file(): raise StepError(step, f"{d}: missing sample {sample}; run the sample step first")
            weights = self.flow_path(d); report = weights.with_name(f"{d}_training.json")
            cmd = [sys.executable, HERE / "train_flow.py", sample, "--dtype", "uint64", "--output", weights, "--sample", self.a.flow_sample, "--steps", self.a.flow_steps, "--monotone"]
            if self.cached_run(step, report, cmd) or "dataset" not in read_json(report):
                r = read_json(report); r.update(dataset=d, source=self.rel(sample), weights=self.rel(weights)); write_json(report, r)
            if not weights.is_file(): raise StepError(step, f"{d}: trainer did not write {weights}")
            r = read_json(report); reports.append(r)
            self.log(step, f"{d}: nll {r.get('best_nll', float('nan')):.4f}, tail conflicts {r.get('tail_conflict_degree_raw')} -> {r.get('tail_conflict_degree_transformed')}, unordered pairs {r.get('unordered_transformed_pairs')}")
        write_json(self.results / "flows/training_report.json", reports)

    def hardness_jobs(self, d: str) -> list[tuple[str, pathlib.Path, list]]:
        full, sample, flow, out = self.dataset_path(d), self.sample_path(d), self.flow_path(d), self.results / "hardness"
        common = ["--dtype", "uint64", "--region-keys", self.a.region_keys, "--pla-eps", "32,4096", "--check-sorted", 1]
        return [("full", out / f"{d}_full.json", [self.hardness_bin, "--data", full, "--flow", flow, *common]),
                ("sample_flow", out / f"{d}_sample_flow.json", [self.hardness_bin, "--data", sample, "--flow", flow, "--virtual-alpha", self.a.alpha, *common]),
                ("sample_csv", out / f"{d}_sample_csv.json", [self.hardness_bin, "--data", sample, "--virtual-alpha", self.a.alpha, *common])]

    def step_hardness(self) -> None:
        step = "hardness"; self.need_binary(step, self.hardness_bin, "scaleli_hardness")
        for d in self.datasets:
            for p in (self.dataset_path(d), self.sample_path(d), self.flow_path(d)):
                if not p.is_file(): raise StepError(step, f"{d}: missing {p}; run the earlier steps first")
        jobs = [(d, kind, out, cmd) for d in self.datasets for kind, out, cmd in self.hardness_jobs(d)]
        jobs.sort(key=lambda j: (j[1] != "full", self.datasets.index(j[0])))   # full-dataset runs first: they dominate the wall clock
        def one(job):
            d, kind, out, cmd = job; self.cached_run(step, out, cmd); return d, kind
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.a.jobs)) as pool:
            for _ in pool.map(one, jobs): pass
        H = read_json(self.results / "hardness.json") if (self.results / "hardness.json").is_file() else {}
        details = read_json(self.results / "hardness_details.json") if (self.results / "hardness_details.json").is_file() else {}
        FLOW, CSV = ("duplicates", "unordered_pairs"), ("virtual_points", "regions")   # degeneracy indicators kept next to the metrics
        for d in self.datasets:
            raw = {kind: read_json(out) for kind, out, _ in self.hardness_jobs(d)}
            scopes = {"full": metric_block(raw["full"].get("original")), "full_flow": metric_block(raw["full"].get("transformed"), FLOW),
                      "sample": metric_block(raw["sample_flow"].get("original")), "sample_flow": metric_block(raw["sample_flow"].get("transformed"), FLOW),
                      "sample_flow_csv": metric_block(raw["sample_flow"].get("smoothed"), CSV), "sample_csv": metric_block(raw["sample_csv"].get("smoothed"), CSV)}
            missing = [s for s, b in scopes.items() if b is None]
            if missing: self.log(step, f"{d}: WARNING scopes without a metric block: {missing}")
            H[d] = {s: b for s, b in scopes.items() if b is not None}
            info = lambda r, keys: {k: r.get(k) for k in keys if k in r}
            details[d] = {"full": info(raw["full"], ["data", "keys", "sorted", "duplicates", "min", "max", "sort_ns", "elapsed_ns"]),
                          "full_original": info(raw["full"].get("original") or {}, ["fmcd"]),
                          "full_flow": info(raw["full"].get("transformed") or {}, ["duplicates", "unordered_pairs", "transform_ns", "fmcd"]),
                          "sample": info(raw["sample_flow"], ["data", "keys", "sorted", "duplicates", "min", "max", "elapsed_ns"]),
                          "sample_flow": info(raw["sample_flow"].get("transformed") or {}, ["duplicates", "unordered_pairs", "transform_ns", "fmcd"]),
                          "sample_flow_csv": info(raw["sample_flow"].get("smoothed") or {}, ["virtual_points", "smoothing_ns", "regions", "fmcd"]),
                          "sample_csv": info(raw["sample_csv"].get("smoothed") or {}, ["virtual_points", "smoothing_ns", "regions", "fmcd"]),
                          "alpha": self.a.alpha, "region_keys": self.a.region_keys,
                          "cd_note": "CD on transformed/smoothed features uses U_T + 1e-6 * mean gap (fmcd.ut_epsilon), not LIPP's absolute 1e-6; "
                                     "fmcd.conflict_degree_lipp_epsilon is the literal-constant value. CD carries +-1 rounding uncertainty (floor of a double product, as in LIPP)."}
            f = raw["full"]
            if f.get("sorted") is False or f.get("duplicates"): self.log(step, f"{d}: WARNING full file sorted={f.get('sorted')} duplicates={f.get('duplicates')}")
            for s in ("full_flow", "sample_flow"):
                b = H[d].get(s) or {}
                if b.get("duplicates") and b.get("keys") and b["duplicates"] > 0.5 * b["keys"]: self.log(step, f"{d}: WARNING {s}: the flow collapses {b['duplicates']:,} of {b['keys']:,} keys onto duplicate feature values; its metrics describe a degenerate sequence")
            self.log(step, f"{d}: " + ", ".join(f"{s} CD={b['conflict_degree']:.0f} PLA-32={b['pla_32']:.0f}" for s, b in H[d].items()))
        write_json(self.results / "hardness.json", H); write_json(self.results / "hardness_details.json", details)

    def sweep_config(self) -> dict:
        common = {"load-ratio": 1, "miss": 0, "query-distribution": "uniform", "ops": self.a.ops, "warmup": self.a.warmup,
                  "verify": 0, "instrument": 1, "latency": 0, "region-keys": self.a.region_keys}   # contract C5 (region-keys mirrors the hardness runs)
        if self.a.build_threads: common["build-threads"] = self.a.build_threads   # optional: only the bulk-load wall clock changes, queries stay single-threaded
        return {"description": "AIDB 2026 hardness-protocol lane, generated by tools/aidb_pipeline.py. " + LABEL,
                "datasets": [{"name": d, "data": self.rel(self.sample_path(d)), "format": "sosd", "dtype": "uint64",
                              "flow_weights": self.rel(self.flow_path(d)), "flow_train_args": ["--monotone"]} for d in self.datasets],
                "profiles": ["read_only"], "variants": self.variants, "seeds": list(self.a.seeds), "repeats": 1, "common": common}

    def step_sweep(self) -> None:
        step = "sweep"; self.need_binary(step, self.bench, "scaleli_bench")
        for g in self.a.extra_variants:
            if not self.bench_accepts(EXTRA_VARIANTS[g]["requires"]):
                raise StepError(step, f"{self.bench} does not accept {EXTRA_VARIANTS[g]['requires']} (needed by --extra-variants {g}); rebuild scaleli_bench from the current src/benchmark.cpp or drop the option")
        for d in self.datasets:
            for p in (self.sample_path(d), self.flow_path(d)):
                if not p.is_file(): raise StepError(step, f"{d}: missing {p}; run the earlier steps first")
        cfg = self.sweep_config(); results, summary, saved = self.sweep_dir / "results.jsonl", self.sweep_dir / "summary.csv", self.sweep_dir / "config.json"
        identity_path = self.sweep_dir / "identity.json"   # what results.jsonl was produced with: the config and the sha256 of scaleli_bench (its path is informational only)
        identity = {"config": cfg, "scaleli_bench_sha256": self.file_hash(self.bench), "scaleli_bench": str(self.bench)}
        same = lambda old: old.get("config") == cfg and old.get("scaleli_bench_sha256") == identity["scaleli_bench_sha256"]
        if self.a.force and self.sweep_dir.exists(): shutil.rmtree(self.sweep_dir); self.log(step, "force: removed previous sweep")
        if results.is_file() and summary.is_file():
            if identity_path.is_file():
                old = read_json(identity_path)
                if same(old): self.log(step, f"keep {self.rel(results)} (same settings and scaleli_bench sha256 {identity['scaleli_bench_sha256'][:16]}…, built as {old.get('scaleli_bench')})"); return
                what = "settings" if old.get("config") != cfg else f"scaleli_bench binary (sha256 {str(old.get('scaleli_bench_sha256'))[:16]}… at {old.get('scaleli_bench')} vs {identity['scaleli_bench_sha256'][:16]}… at {self.bench})"
                raise StepError(step, f"{self.sweep_dir} holds a sweep produced with different {what}; pass --force or use another --results")
            if saved.is_file() and read_json(saved) == cfg:
                self.log(step, f"keep {self.rel(results)} (WARNING: sweep predates binary fingerprinting, the scaleli_bench that produced it is unknown; --force to redo)"); return
            self.log(step, f"resuming incomplete sweep in {self.rel(results)} (no identity.json yet); run_suite --resume appends the missing jobs")
        elif results.is_file():
            self.log(step, f"resuming incomplete sweep in {self.rel(results)}; run_suite --resume appends the missing jobs")
        if results.is_file():
            stale = results.with_name(f"results.partial-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"); results.replace(stale); self.log(step, f"incomplete sweep moved to {stale.name}")
        write_json(self.config_path, cfg); self.sweep_dir.mkdir(parents=True, exist_ok=True); write_json(saved, cfg); write_json(identity_path, identity)
        for p in ("environment.json", "failed_command.json"):
            if (self.sweep_dir / p).exists(): (self.sweep_dir / p).unlink()
        self.run(step, [sys.executable, HERE / "run_suite.py", "--binary", self.bench, "--config", self.config_path, "--output", self.sweep_dir, "--timeout", int(self.a.timeout), "--resume"])
        self.run(step, [sys.executable, HERE / "summarize.py", results, "--baseline", "packed_rank", "--output", summary])
        self.run(step, [sys.executable, HERE / "sweep_report.py", summary, "--output", self.sweep_dir / "sweep_report.html"])

    def step_throughput(self) -> None:
        step = "throughput"; results, out = self.sweep_dir / "results.jsonl", self.results / "throughput.json"
        if not results.is_file(): raise StepError(step, f"missing {results}; run the sweep step first")
        if fresh(out, results) and not self.a.force: self.log(step, f"keep {out.name}"); return
        rows = [json.loads(s) for s in results.read_text().splitlines() if s.strip()]
        if not rows: raise StepError(step, f"{results} is empty")
        T, detail = median_throughput(rows); write_json(out, T); write_json(self.results / "throughput_detail.json", detail)
        for v, per in T.items(): self.log(step, f"{v}: " + ", ".join(f"{d} {m:.2f}" for d, m in per.items()) + " MOPS")

    def step_scores(self) -> None:
        step = "scores"; H, T, out = self.results / "hardness.json", self.results / "throughput.json", self.results / "scores.json"
        for p in (H, T):
            if not p.is_file(): raise StepError(step, f"missing {p}; run the hardness and throughput steps first")
        if fresh(out, H, T, HERE / "aidb_scores.py") and not self.a.force: self.log(step, f"keep {out.name}"); return
        self.run(step, [sys.executable, HERE / "aidb_scores.py", "--hardness", H, "--throughput", T, "--output", out])
        S = read_json(out)
        for scope, block in S.items():
            m = block.get("metrics", {}) if isinstance(block, dict) else {}
            picks = [k for k in ("PLA-32", "CD", "PLA-32·PLA-4096", "CD·PLA-32") if k in m]
            skipped = block.get("skipped") or {}
            self.log(step, f"{scope}: " + ", ".join(f"{k} Conf={fmt(m[k].get('conformance'))} Cov={fmt(m[k].get('coverage'))}" for k in picks)
                     + (f"; skipped variants: {', '.join(f'{v} ({why})' for v, why in skipped.items())}" if skipped else ""))

    def step_report(self) -> None:
        step = "report"; out = self.results / "report.html"
        inputs = [self.results / p for p in ("hardness.json", "throughput.json", "scores.json", "provenance.json", "sweep/summary.csv", "flows/training_report.json")] + [HERE / "aidb_report.py"]
        explicit = self.a.steps is not None and self.a.step_list == ["report"]   # `--steps report` alone always regenerates the page
        if not explicit and fresh(out, *inputs) and not self.a.force: self.log(step, f"keep {out.name}"); return
        self.run(step, [sys.executable, HERE / "aidb_report.py", "--results", self.results, "--output", out])
        self.log(step, f"wrote {out}")

    # ----- driver --------------------------------------------------------------------------
    def run_record(self, steps: list[str], started: str, status: str) -> dict:
        a = self.a
        return {"label": LABEL, "started_utc": started, "finished_utc": utc_now(), "status": status, "dry_run": self.dry, "steps": steps, "datasets": self.datasets,
                "settings": {"sample": a.sample, "sample_seed": a.sample_seed, "sample_mode": a.sample_mode, "ops": a.ops, "warmup": a.warmup, "seeds": list(a.seeds), "alpha": a.alpha,
                             "region_keys": a.region_keys, "flow_sample": a.flow_sample, "flow_steps": a.flow_steps, "jobs": a.jobs, "force": a.force,
                             "build_threads": a.build_threads, "extra_variants": list(a.extra_variants)},
                "variants": [v["name"] for v in self.variants],
                "paths": {"data_dir": str(self.data_dir), "samples_dir": str(self.samples_dir), "results": str(self.results), "binary_dir": str(self.binary_dir), "config": str(self.config_path)},
                "binaries": {p.name: self.file_hash(p) for p in (self.bench, self.hardness_bin) if p.is_file()},
                "scale_note": "Hardness metrics use the full files (sorted copies for the seven GRE files served unsorted); index runs use uniform samples; "
                              "paper protocol is 200M keys, 20M warm-up and 100M measured lookups per run."}

    def execute(self, steps: list[str]) -> None:
        started = utc_now(); self.log("pipeline", f"start steps={steps} datasets={self.datasets} results={self.results} dry_run={self.dry} variants={[v['name'] for v in self.variants]}")
        for s in steps:
            t0 = time.perf_counter(); self.log(s, "begin"); getattr(self, "step_" + s.replace("-", "_"))(); self.log(s, f"done in {time.perf_counter() - t0:.1f} s")
        write_json(self.results / "run.json", self.run_record(steps, started, "finished"))   # written only once every requested step succeeded
        self.log("pipeline", "finished")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", metavar="NAME", help=f"subset of {' '.join(GRE_DATASETS)} (dry run: {' '.join(DRY_DATASETS)})")
    p.add_argument("--data-dir", type=pathlib.Path, help="directory holding the SOSD files <name> (default data/external/gre; dry run: <results>/data)")
    p.add_argument("--samples-dir", type=pathlib.Path, help="where samples go (default data/samples; dry run: <results>/samples)")
    p.add_argument("--build-threads", type=int, help="optional scaleli_bench --build-threads for the bulk load (regions build in parallel; queries stay single-threaded); omitted from the sweep config unless given")
    p.add_argument("--extra-variants", nargs="*", default=[], choices=sorted(EXTRA_VARIANTS), metavar="GROUP",
                   help="optional variant groups beyond the eight contracted ones: " + ", ".join(f"{g} ({', '.join(v['name'] for v in spec['variants'])}; needs scaleli_bench {spec['requires']})" for g, spec in EXTRA_VARIANTS.items()))
    p.add_argument("--sample-mode", default="uniform", choices=["uniform", "window", "strided"], help="uniform keeps global shape but flattens local structure; window keeps local structure of one contiguous key range (seeded offset)")
    p.add_argument("--results", type=pathlib.Path, help="output directory (default results/aidb; dry run: results/aidb_dry)")
    p.add_argument("--sample", type=int, help="uniform sample size for the index runs (default 2000000; dry run 20000)")
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--ops", type=int, help="measured lookups per run (default 1000000; dry run 20000)")
    p.add_argument("--warmup", type=int, help="warm-up lookups per run (default 200000; dry run 2000)")
    p.add_argument("--seeds", help="comma-separated workload seeds (default 11,29,47; dry run 11,29)")
    p.add_argument("--alpha", type=float, default=0.1, help="CSV-style virtual-point budget per region")
    p.add_argument("--region-keys", type=int, default=4096)
    p.add_argument("--flow-sample", type=int, default=4096, help="training keys for tools/train_flow.py")
    p.add_argument("--flow-steps", type=int, default=200)
    p.add_argument("--binary-dir", type=pathlib.Path, default=SOTA / "build/experimental/scaleli", help="directory with scaleli_bench and scaleli_hardness")
    p.add_argument("--jobs", type=int, default=1, help="concurrent scaleli_hardness processes (each full run holds the whole file in memory)")
    p.add_argument("--wait", action="store_true", help="poll until every requested dataset finished downloading (never downloads)")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--timeout", type=float, default=4 * 3600, help="per-subprocess timeout in seconds")
    p.add_argument("--force", action="store_true", help="redo steps even when their outputs exist")
    p.add_argument("--dry-run", action="store_true", help="fixture + four synthetic sets (one shuffled with duplicates), tiny settings, results/aidb_dry, no network")
    p.add_argument("--steps", help="comma-separated subset of: " + " ".join(STEPS) + " (`--steps report` alone always regenerates report.html)")
    a = p.parse_args(argv)
    dry = a.dry_run
    a.results = a.results or (ROOT / ("results/aidb_dry" if dry else "results/aidb"))
    a.data_dir = a.data_dir or (a.results / "data" if dry else ROOT / "data/external/gre")
    a.samples_dir = a.samples_dir or (a.results / "samples" if dry else ROOT / "data/samples")
    a.sample = a.sample or (20000 if dry else 2000000); a.ops = a.ops or (20000 if dry else 1000000); a.warmup = a.warmup if a.warmup is not None else (2000 if dry else 200000)
    a.seeds = [int(x) for x in (a.seeds or ("11,29" if dry else "11,29,47")).split(",") if x.strip()]
    a.datasets = a.datasets or (list(DRY_DATASETS) if dry else list(GRE_DATASETS))
    known = DRY_DATASETS if dry else GRE_DATASETS
    unknown = [d for d in a.datasets if d not in known]
    if unknown: p.error(f"unknown dataset(s) {unknown}; choose from {list(known)}")
    steps = [s.strip() for s in (a.steps or ",".join(STEPS)).split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad: p.error(f"unknown step(s) {bad}; choose from {STEPS}")
    a.step_list = [s for s in STEPS if s in steps]
    a.extra_variants = list(dict.fromkeys(a.extra_variants))
    if a.sample < 1 or a.ops < 1 or a.warmup < 0 or not a.seeds or not (0 <= a.alpha < 1) or a.region_keys < 2 or a.jobs < 1 or (a.build_threads is not None and a.build_threads < 1): p.error("invalid numeric argument")
    return a


def main(argv=None) -> int:
    a = parse_args(argv); pipe = Pipeline(a)
    try: pipe.execute(a.step_list)
    except StepError as e:
        pipe.log(e.step, f"FAILED: {e}"); print(f"FAILED step {e.step}: {e}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        pipe.log("pipeline", "interrupted"); return 130
    print(json.dumps({"results": str(pipe.results), "steps": a.step_list, "datasets": pipe.datasets, "variants": [v["name"] for v in pipe.variants],
                      "report": str(pipe.results / "report.html"), "log": str(pipe.log_path), "label": LABEL}, indent=2)); return 0


if __name__ == "__main__": sys.exit(main())
