"""tools/aidb_pipeline.py helpers and settings that need no binaries: contract C5 variant list, opt-in extras,
content-aware JSON writes, SOSD helpers, stale-sample reasons, cached-run identities. The end-to-end behaviour
(sort step on an unsorted file with duplicates, provenance, sweep, scores, report) is exercised by --dry-run."""
from __future__ import annotations
import array, importlib.util, json, os, pathlib, sys, tempfile, time, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aidb_pipeline", ROOT / "tools" / "aidb_pipeline.py"); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
CONTRACT = ["sorted_vector", "raw_rank", "packed_rank", "packed_rank_flow", "packed_rank_flow_forced", "packed_rank_vp10", "packed_rank_flow_vp10", "packed_byte"]


class Settings(unittest.TestCase):
    def pipeline(self, tmp, *extra):
        a = P.parse_args(["--dry-run", "--results", str(tmp / "r"), "--binary-dir", str(tmp / "nobin"), *extra]); pipe = P.Pipeline(a)
        pipe.data_dir.mkdir(parents=True, exist_ok=True)
        for name in P.DRY_DATASETS: P.write_sosd(pipe.raw_path(name), array.array("Q", range(1, 40001)))  # sweep_config() reads the key counts
        return pipe

    def test_contract_variants_and_common_block(self):
        with tempfile.TemporaryDirectory() as d:
            pipe = self.pipeline(pathlib.Path(d)); cfg = pipe.sweep_config()
            self.assertEqual([v["name"] for v in cfg["variants"]], CONTRACT)
            self.assertEqual(cfg["common"], {"load-ratio": 1, "miss": 0, "query-distribution": "uniform", "ops": 20000, "warmup": 2000, "verify": 0, "instrument": 1, "latency": 0, "region-keys": 4096})
            self.assertEqual(cfg["profiles"], ["read_only"]); self.assertEqual(cfg["seeds"], [11, 29])
            self.assertEqual(len(P.STEPS), 9); self.assertIn("sort", P.STEPS); self.assertIn("shuffled_dups", P.DRY_DATASETS)

    def test_extra_variants_and_build_threads_are_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            pipe = self.pipeline(pathlib.Path(d), "--extra-variants", "fusion", "--build-threads", "4"); cfg = pipe.sweep_config()
            self.assertEqual([v["name"] for v in cfg["variants"]], CONTRACT + ["packed_rank_fusion_auto", "packed_rank_flow_costsel"])
            self.assertTrue(all(v.get("fusion") == "auto" for v in cfg["variants"][8:]))
            self.assertEqual(cfg["common"]["build-threads"], 4)
            self.assertEqual(P.EXTRA_VARIANTS["fusion"]["requires"], "--fusion")
            with self.assertRaises(SystemExit): P.parse_args(["--dry-run", "--extra-variants", "nope"])
            with self.assertRaises(SystemExit): P.parse_args(["--dry-run", "--build-threads", "0"])

    def test_dry_dataset_urls_and_report_step_detection(self):
        with tempfile.TemporaryDirectory() as d:
            pipe = self.pipeline(pathlib.Path(d))
            self.assertTrue(pipe.url("dense_sparse_fixture").startswith("file://"))
            self.assertIn("shuffled=1&duplicates=500", pipe.url("shuffled_dups")); self.assertIn("seed=7", pipe.url("shuffled_dups"))
            self.assertEqual(P.parse_args(["--dry-run", "--steps", "report"]).step_list, ["report"])
            self.assertEqual(P.parse_args(["--dry-run", "--steps", "report,sort"]).step_list, ["sort", "report"])
            self.assertIsNone(P.parse_args(["--dry-run"]).steps)

    def test_median_throughput_orders_contract_variants_first(self):
        rows = [{"dataset": "x", "profile": "read_only", "seed": s, "repeat": 0, "trace_fingerprint": "f" + str(s), "variant": v, "throughput_ops_s": m * 1e6}
                for s in (1, 2) for v, m in (("packed_byte", 3.0), ("zzz_custom", 9.0), ("sorted_vector", 5.0), ("packed_rank_fusion_auto", 4.0))]
        T, detail = P.median_throughput(rows)
        self.assertEqual(list(T), ["sorted_vector", "packed_byte", "packed_rank_fusion_auto", "zzz_custom"])
        self.assertEqual(T["packed_byte"]["x"], 3.0); self.assertEqual(len(detail["zzz_custom"]["x"]), 2)
        rows[0]["trace_fingerprint"] = "other"
        with self.assertRaises(P.StepError): P.median_throughput(rows)


class Helpers(unittest.TestCase):
    def test_write_json_leaves_identical_files_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "a" / "x.json"
            self.assertTrue(P.write_json(p, {"k": [1, 2]})); m = p.stat().st_mtime_ns
            os.utime(p, ns=(m - 5_000_000_000, m - 5_000_000_000)); m = p.stat().st_mtime_ns
            self.assertFalse(P.write_json(p, {"k": [1, 2]})); self.assertEqual(p.stat().st_mtime_ns, m)
            self.assertTrue(P.write_json(p, {"k": [1, 3]})); self.assertNotEqual(p.stat().st_mtime_ns, m)

    def test_sosd_helpers_and_metric_block(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "k.sosd"
            P.write_sosd(p, array.array("Q", [5, 1, 9, 9])); self.assertEqual(P.sosd_count(p), 4); self.assertEqual(list(P.sosd_keys(p)), [5, 1, 9, 9])
            self.assertFalse(P.sosd_strictly_ascending(p))
            P.write_sosd(p, array.array("Q", [1, 5, 9, 2 ** 64 - 1])); self.assertTrue(P.sosd_strictly_ascending(p))
            self.assertEqual(P.size_label(2_000_000), "2M"); self.assertEqual(P.size_label(16384), "16384")
        raw = {"rmse": 1, "max_error": 2, "conflict_degree": 3, "pla": {"32": 4, "4096": 5}, "keys": 10, "duplicates": 7, "unordered_pairs": 0, "transform_ns": 99}
        self.assertEqual(P.metric_block(raw), {"rmse": 1.0, "max_error": 2.0, "conflict_degree": 3.0, "pla_32": 4.0, "pla_4096": 5.0, "keys": 10})
        self.assertEqual(P.metric_block(raw, ("duplicates", "unordered_pairs"))["duplicates"], 7)
        self.assertNotIn("transform_ns", P.metric_block(raw, ("duplicates", "unordered_pairs")))
        self.assertIsNone(P.metric_block(None))
        with self.assertRaises(P.StepError): P.metric_block({"rmse": 1})

    def test_sample_stale_reason_and_cached_run_identity(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d); a = P.parse_args(["--dry-run", "--results", str(tmp / "r"), "--binary-dir", str(tmp / "nobin"), "--sample", "3"]); pipe = P.Pipeline(a)
            src = pipe.raw_path("lognormal"); src.parent.mkdir(parents=True); P.write_sosd(src, array.array("Q", [1, 2, 3, 4, 5]))
            dest = pipe.sample_path("lognormal"); manifest = dest.with_name(dest.name + ".manifest.json"); dest.parent.mkdir(parents=True)
            self.assertEqual(pipe.sample_stale_reason("lognormal", src, dest, manifest, {}), "missing")
            P.write_sosd(dest, array.array("Q", [1, 3, 5])); entry = {"sha256": "abc"}
            good = {"source": str(src), "source_sha256": "abc", "sample_count": 3, "mode": "uniform", "seed": 42}
            manifest.write_text(json.dumps(good)); self.assertIsNone(pipe.sample_stale_reason("lognormal", src, dest, manifest, entry))
            manifest.write_text(json.dumps(dict(good, source_sha256="zzz"))); self.assertIn("sha256", pipe.sample_stale_reason("lognormal", src, dest, manifest, entry))
            manifest.write_text(json.dumps(dict(good, source="/elsewhere"))); self.assertIn("drawn from /elsewhere", pipe.sample_stale_reason("lognormal", src, dest, manifest, entry))
            manifest.write_text(json.dumps(dict(good, seed=1))); self.assertIn("seed", pipe.sample_stale_reason("lognormal", src, dest, manifest, entry))
            manifest.write_text(json.dumps(good)); P.write_sosd(dest, array.array("Q", [3, 1, 5])); self.assertIn("ascending", pipe.sample_stale_reason("lognormal", src, dest, manifest, entry))
            # the sorted copy's hash is the reference once a .sorted file is in use
            sorted_copy = pipe.sorted_path("lognormal"); P.write_sosd(sorted_copy, array.array("Q", [1, 2, 3, 4, 5]))
            self.assertEqual(pipe.source_sha256("lognormal", sorted_copy, {"sha256": "abc", "sort_audit": {"sha256": "sorted!"}}), "sorted!")
            self.assertEqual(pipe.source_sha256("lognormal", src, {"sha256": "abc"}), "abc")
            # cached_run records the sha256 of the executable and script, keeps on identical identity, warns on legacy list side files
            script = tmp / "emit.py"; script.write_text("import json,sys; print(json.dumps({'arg': sys.argv[1]}))")
            out = tmp / "r" / "cached" / "o.json"; cmd = [sys.executable, script, "v1"]
            self.assertTrue(pipe.cached_run("t", out, cmd)); side = json.loads(out.with_name("o.json.command.json").read_text())
            self.assertEqual(set(side["sha256"]), {sys.executable, str(script)}); self.assertEqual(side["command"][2], "v1")
            self.assertFalse(pipe.cached_run("t", out, cmd))
            script.write_text("import json,sys; print(json.dumps({'arg': sys.argv[1], 'v': 2}))"); pipe._hashes.clear()
            self.assertTrue(pipe.cached_run("t", out, cmd)); self.assertEqual(json.loads(out.read_text())["v"], 2)
            self.assertIn("redo cached/o.json: same arguments, different binary/script sha256", pipe.log_path.read_text())
            twin = tmp / "twin.py"; twin.write_bytes(script.read_bytes())  # byte-identical script at another path: same identity, kept
            self.assertFalse(pipe.cached_run("t", out, [sys.executable, twin, "v1"]))
            self.assertTrue(pipe.cached_run("t", out, [sys.executable, twin, "v2"]))  # different argument: rerun
            out.with_name("o.json.command.json").write_text(json.dumps([str(c) for c in cmd]))
            self.assertFalse(pipe.cached_run("t", out, cmd)); self.assertIn("predates binary fingerprinting", pipe.log_path.read_text())
            self.assertTrue(pipe.cached_run("t", out, cmd, force=True))


if __name__ == "__main__": unittest.main()
