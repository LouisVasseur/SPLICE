"""Tests for tools/aidb_scores.py: AIDB 2026 conformance/coverage protocol (sec. 3.2) on our control variants."""
from __future__ import annotations
import importlib.util, itertools, json, math, pathlib, random, subprocess, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "aidb_scores.py"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = load("aidb_scores")
DOT = S.SEP


def hardness_from(**cols):
    """Build {dataset: {"sample": {five fields}}} from per-metric column dicts; missing metrics get 1.0."""
    names = set().union(*[set(v) for v in cols.values()])
    out = {}
    for d in sorted(names):
        row = {field: 1.0 for field in S.FIELDS.values()}
        for metric, values in cols.items():
            row[S.FIELDS[metric]] = values[d]
        out[d] = {"sample": row}
    return out


class HandBuiltExample(unittest.TestCase):
    """Four datasets a,b,c,d, one variant; every number below is worked by hand in the comments."""
    # Hardness (larger = harder):  RMSE  a:1 b:2 c:3 d:4      ME  a:1 b:3 c:2 d:4      CD  all 1 (ties)
    # Throughput p (MOPS):         a:4 b:2 c:3 d:1
    # mean p = 2.5; population variance = ((1.5)^2+(0.5)^2+(0.5)^2+(1.5)^2)/4 = 5/4; std = sqrt(1.25) = 1.118034
    # p_hat = p/std:  a 3.57771  b 1.78885  c 2.68328  d 0.89443
    # RMSE orders every pair; (harder, easier) pairs and gap = p_hat(harder) - p_hat(easier):
    #   (b,a) -1.78885 conforming  w = sigmoid(-1.78885) = 0.14321
    #   (c,a) -0.89443 conforming  w = 0.29020
    #   (d,a) -2.68328 conforming  w = 0.06397
    #   (c,b) +0.89443 VIOLATING   w = sigmoid(+0.89443) = 0.70980   (c is harder but faster than b)
    #   (d,b) -0.89443 conforming  w = 0.29020
    #   (d,c) -1.78885 conforming  w = 0.14321
    #   R = 0.14321+0.29020+0.06397+0.29020+0.14321 = 0.93079 ; P = 0.70980
    #   Conf_RMSE = (0.93079-0.70980)/(0.93079+0.70980) = 0.22099/1.64059 = 0.13470 ; Cov_RMSE = (6-0)/6 = 1
    # RMSE·ME: b vs c is incomparable (RMSE 2<3 but ME 3>2), the other five pairs keep their RMSE direction:
    #   |C| = 5, |U| = 1 -> Cov = (5-1)/6 = 0.66667 ; the violating pair (c,b) is gone -> P = 0, Conf = 1
    # RMSE·ME·CD: CD ties on every pair, so >= holds and comparability is decided by RMSE·ME -> same numbers.
    # CD alone: every pair tied on every dimension -> all 6 incomparable: Cov = (0-6)/6 = -1, R+P = 0 -> Conf = 0.
    H = hardness_from(RMSE={"a": 1, "b": 2, "c": 3, "d": 4}, ME={"a": 1, "b": 3, "c": 2, "d": 4})
    T = {"packed_rank": {"a": 4.0, "b": 2.0, "c": 3.0, "d": 1.0}}

    def scores(self, **kw):
        return S.score_all(self.H, self.T, **kw)["sample"]

    def test_normalization(self):
        p_hat = S.normalize(self.T["packed_rank"])
        self.assertAlmostEqual(S.std([4, 2, 3, 1]), math.sqrt(1.25), places=12)
        for d, expect in {"a": 3.57771, "b": 1.78885, "c": 2.68328, "d": 0.89443}.items():
            self.assertAlmostEqual(p_hat[d], expect, places=4)

    def test_scalar_rmse(self):
        m = self.scores()["metrics"]["RMSE"]
        self.assertEqual(m["comparable_pairs"], 6)
        self.assertEqual(m["incomparable_pairs"], 0)
        self.assertEqual(m["coverage"], 1.0)
        detail = m["per_variant_detail"]["packed_rank"]
        self.assertEqual((detail["conforming"], detail["violating"]), (5, 1))
        self.assertAlmostEqual(detail["reward"], 0.93079, places=4)
        self.assertAlmostEqual(detail["penalty"], 0.70980, places=4)
        self.assertAlmostEqual(m["per_variant"]["packed_rank"], 0.13470, places=4)
        self.assertAlmostEqual(m["conformance"], 0.13470, places=4)
        self.assertIn(["c", "b"], m["pairs"]["comparable"])

    def test_two_dim_rmse_me_trades_coverage_for_conformance(self):
        m = self.scores()["metrics"]["RMSE" + DOT + "ME"]
        self.assertEqual((m["comparable_pairs"], m["incomparable_pairs"]), (5, 1))
        self.assertAlmostEqual(m["coverage"], 4 / 6, places=12)
        self.assertEqual(m["pairs"]["incomparable"], [["b", "c"]])
        self.assertAlmostEqual(m["per_variant_detail"]["packed_rank"]["reward"], 0.93079, places=4)
        self.assertEqual(m["per_variant_detail"]["packed_rank"]["penalty"], 0.0)
        self.assertEqual(m["conformance"], 1.0)

    def test_three_dim_with_tied_dimension_matches_two_dim(self):
        block = self.scores()["metrics"]
        two, three = block["RMSE" + DOT + "ME"], block["RMSE" + DOT + "ME" + DOT + "CD"]
        self.assertEqual(three["pairs"], two["pairs"])
        self.assertAlmostEqual(three["conformance"], two["conformance"], places=12)
        self.assertAlmostEqual(three["coverage"], two["coverage"], places=12)

    def test_all_ties_are_incomparable(self):
        m = self.scores()["metrics"]["CD"]
        self.assertEqual((m["comparable_pairs"], m["incomparable_pairs"]), (0, 6))
        self.assertEqual(m["coverage"], -1.0)
        self.assertEqual(m["per_variant"]["packed_rank"], 0.0)  # R + P == 0 -> defined as 0
        self.assertEqual(m["conformance"], 0.0)

    def test_sample_std_changes_scale_not_sign(self):
        # ddof=1: std = sqrt(5/3) = 1.29099; gaps shrink, so the lone violating pair weighs less
        # relative to the conforming ones only through the sigmoid: the sign of Conf is unchanged.
        pop, sample = self.scores(ddof=0), self.scores(ddof=1)
        self.assertEqual(sample["ddof"], 1)
        self.assertAlmostEqual(sample["normalized_throughput"]["packed_rank"]["a"], 4 / math.sqrt(5 / 3), places=9)
        self.assertNotAlmostEqual(pop["metrics"]["RMSE"]["conformance"], sample["metrics"]["RMSE"]["conformance"], places=3)
        self.assertGreater(sample["metrics"]["RMSE"]["conformance"], 0.0)


class ProtocolProperties(unittest.TestCase):
    def random_scope(self, seed: int, n: int = 8):
        rng = random.Random(seed)
        names = [f"d{i}" for i in range(n)]
        cols = {metric: {d: rng.uniform(1, 1000) for d in names} for metric in S.SCALARS}
        for values in cols.values():  # distinct by construction with continuous draws; assert anyway
            self.assertEqual(len(set(values.values())), n)
        throughput = {"v": {d: rng.uniform(1, 50) for d in names}}
        return S.score_all(hardness_from(**cols), throughput)["sample"]["metrics"]

    def test_scalar_metric_has_full_coverage_when_values_are_distinct(self):
        for seed in range(20):
            metrics = self.random_scope(seed)
            for name in S.SCALARS:
                self.assertEqual(metrics[name]["coverage"], 1.0, name)
                self.assertEqual(metrics[name]["incomparable_pairs"], 0)

    def test_adding_a_dimension_never_increases_coverage(self):
        # With distinct scalar values a pair comparable under A·B is comparable under A, and a pair
        # comparable under A·B·C is comparable under A·B, so coverage is monotone non-increasing.
        # (Ties are the exception: a tie on A can be broken by B, so the test uses distinct values.)
        for seed in range(30):
            metrics = self.random_scope(seed)
            for pair in itertools.combinations(S.SCALARS, 2):
                self.assertLessEqual(metrics[DOT.join(pair)]["coverage"], metrics[pair[0]]["coverage"])
                self.assertLessEqual(metrics[DOT.join(pair)]["coverage"], metrics[pair[1]]["coverage"])
            for triple in itertools.combinations(S.SCALARS, 3):
                for sub in itertools.combinations(triple, 2):
                    self.assertLessEqual(metrics[DOT.join(triple)]["coverage"], metrics[DOT.join(sub)]["coverage"])

    def test_perfectly_conforming_ordering_scores_one_and_reversed_minus_one(self):
        names = [f"d{i}" for i in range(6)]
        cols = {metric: {d: float(i + 1) * (k + 1) for i, d in enumerate(names)} for k, metric in enumerate(S.SCALARS)}
        H = hardness_from(**cols)
        conforming = {"good": {d: 60.0 - 7 * i for i, d in enumerate(names)}}  # harder -> strictly slower
        reversed_ = {"bad": {d: 5.0 + 7 * i for i, d in enumerate(names)}}  # harder -> strictly faster
        both = S.score_all(H, {**conforming, **reversed_})["sample"]["metrics"]
        for name, m in both.items():
            self.assertEqual(m["coverage"], 1.0, name)  # all dimensions agree, so every pair is comparable
            self.assertAlmostEqual(m["per_variant"]["good"], 1.0, places=12, msg=name)
            self.assertAlmostEqual(m["per_variant"]["bad"], -1.0, places=12, msg=name)
            self.assertAlmostEqual(m["conformance"], 0.0, places=12, msg=name)  # mean of +1 and -1

    def test_zero_std_variant_is_skipped_not_scored(self):
        # eq. (1) divides by the std over datasets; a flat variant has std 0 and would otherwise score +1 on
        # every metric (every gap 0, every pair "conforming" with w = 0.5) and lift the mean of eq. (2).
        H = hardness_from(RMSE={"a": 1, "b": 2, "c": 3})
        block = S.score_all(H, {"flat": {"a": 5.0, "b": 5.0, "c": 5.0}, "good": {"a": 3.0, "b": 2.0, "c": 1.0}})["sample"]
        self.assertEqual(block["variants"], ["good"])
        self.assertIn("zero throughput std", block["skipped"]["flat"])
        m = block["metrics"]["RMSE"]
        self.assertNotIn("flat", m["per_variant"])
        self.assertAlmostEqual(m["conformance"], 1.0, places=12)  # good alone: harder is slower on every pair
        alone = S.score_all(H, {"flat": {"a": 5.0, "b": 5.0, "c": 5.0}})["sample"]
        self.assertEqual(alone["variants"], [])
        self.assertEqual(alone["metrics"]["RMSE"]["per_variant"], {})
        self.assertIsNone(alone["metrics"]["RMSE"]["conformance"])
        self.assertEqual(alone["datasets"], ["a", "b", "c"])  # coverage is still reported for the hardness side
        self.assertEqual(alone["metrics"]["RMSE"]["coverage"], 1.0)
        self.assertEqual(S.conformance_of_variant({"a": 0.0, "b": 0.0}, [("b", "a")])["reward"], 0.5)  # the primitive is unchanged

    def test_importance_is_sigmoid_of_signed_gap(self):
        self.assertAlmostEqual(S.sigmoid(0.0), 0.5)
        self.assertAlmostEqual(S.sigmoid(2.0), 1 / (1 + math.exp(-2.0)), places=15)
        self.assertAlmostEqual(S.sigmoid(-2.0), 1 - S.sigmoid(2.0), places=15)
        self.assertEqual(S.sigmoid(1000.0), 1.0)
        self.assertEqual(S.sigmoid(-1000.0), 0.0)
        # a violating pair with a large gap weighs more than one with a small gap; conforming pairs the opposite
        big = S.conformance_of_variant({"h": 3.0, "e": 0.0}, [("h", "e")])
        small = S.conformance_of_variant({"h": 0.5, "e": 0.0}, [("h", "e")])
        self.assertGreater(big["penalty"], small["penalty"])
        big_ok = S.conformance_of_variant({"h": 0.0, "e": 3.0}, [("h", "e")])
        small_ok = S.conformance_of_variant({"h": 0.0, "e": 0.5}, [("h", "e")])
        self.assertLess(big_ok["reward"], small_ok["reward"])

    def test_canonical_metric_names(self):
        names = S.metric_names()
        self.assertEqual(len(names), 25)
        self.assertEqual(names[:5], ["RMSE", "ME", "CD", "PLA-32", "PLA-4096"])
        self.assertEqual(sum(1 for n in names if n.count(DOT) == 1), 10)
        self.assertEqual(sum(1 for n in names if n.count(DOT) == 2), 10)
        self.assertIn("CD" + DOT + "PLA-32", names)
        self.assertIn("RMSE" + DOT + "CD" + DOT + "PLA-32", names)
        self.assertEqual(names[-1], "CD" + DOT + "PLA-32" + DOT + "PLA-4096")
        self.assertEqual(S.ALIASES["PLA-32" + DOT + "PLA-4096"], "GRE")


class DatasetSelection(unittest.TestCase):
    H = hardness_from(RMSE={"a": 1, "b": 2, "c": 3, "d": 4})

    def test_nan_and_inf_throughput_are_dropped_with_a_reason(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            block = S.score_all(self.H, {"ix": {"a": 4.0, "b": bad, "c": 2.0, "d": 1.0}, "iy": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}})["sample"]
            self.assertEqual(block["datasets"], ["a", "c", "d"])
            self.assertEqual(block["variants"], ["ix", "iy"])
            self.assertIn("non-finite throughput for variant ix", block["dropped_datasets"]["b"])
            self.assertIn("removed from every variant", block["dropped_datasets"]["b"])
            for v in ("ix", "iy"):
                self.assertTrue(all(math.isfinite(x) for x in block["normalized_throughput"][v].values()))

    def test_null_throughput_shrinks_the_corpus_and_says_so(self):
        block = S.score_all(self.H, {"ix": {"a": 4.0, "b": None, "c": 2.0, "d": 1.0}, "iy": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}})["sample"]
        self.assertEqual(block["datasets"], ["a", "c", "d"])
        self.assertIn("null throughput for variant ix", block["dropped_datasets"]["b"])
        missing = S.score_all(self.H, {"ix": {"a": 4.0, "c": 2.0, "d": 1.0}, "iy": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}})["sample"]
        self.assertIn("no throughput entry for variant ix", missing["dropped_datasets"]["b"])

    def test_string_throughput_skips_the_variant_with_the_real_reason(self):
        block = S.score_all(self.H, {"ix": {"a": "4.0", "b": "3.0", "c": "2.0", "d": "1.0"}, "iy": {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}})["sample"]
        self.assertEqual(block["variants"], ["iy"])
        self.assertIn("non-numeric or non-finite throughput value(s)", block["skipped"]["ix"])
        self.assertIn("a (non-numeric throughput (str))", block["skipped"]["ix"])
        self.assertEqual(block["datasets"], ["a", "b", "c", "d"])  # a skipped variant does not shrink the corpus

    def test_cli_accepts_nan_in_throughput_json(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "H.json").write_text(json.dumps(self.H))
            (root / "T.json").write_text('{"ix": {"a": 4.0, "b": NaN, "c": 2.0, "d": 1.0}}')
            p = subprocess.run([sys.executable, str(TOOL), "--hardness", str(root / "H.json"), "--throughput", str(root / "T.json"), "--output", str(root / "S.json"), "--print"],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            S_json = json.loads((root / "S.json").read_text())
            self.assertIn("b", S_json["sample"]["dropped_datasets"])
            self.assertIn("dataset b dropped", p.stdout)

    def test_variant_with_fewer_than_three_datasets_is_skipped(self):
        block = S.score_all(self.H, {"full": {"a": 4, "b": 3, "c": 2, "d": 1}, "tiny": {"a": 1, "b": 2}, "alien": {"x": 1, "y": 2, "z": 3}})["sample"]
        self.assertEqual(block["variants"], ["full"])
        self.assertIn("skipped", block["skipped"]["tiny"])
        self.assertIn("skipped", block["skipped"]["alien"])
        self.assertEqual(block["datasets"], ["a", "b", "c", "d"])
        self.assertEqual(set(block["metrics"]["RMSE"]["per_variant"]), {"full"})

    def test_datasets_are_the_intersection_of_hardness_and_every_scored_variant(self):
        block = S.score_all(self.H, {"v1": {"a": 4, "b": 3, "c": 2, "d": 1, "e": 9}, "v2": {"a": 4, "b": 3, "c": 2}})["sample"]
        self.assertEqual(block["datasets"], ["a", "b", "c"])
        self.assertEqual(block["variants"], ["v1", "v2"])
        self.assertEqual(block["metrics"]["RMSE"]["comparable_pairs"], 3)

    def test_scopes_are_independent_and_missing_fields_drop_a_dataset(self):
        H = json.loads(json.dumps(self.H))
        H["a"]["sample_flow"] = dict(H["a"]["sample"], rmse=9.0)
        H["b"]["sample_flow"] = dict(H["b"]["sample"])
        H["c"]["sample_flow"] = dict(H["c"]["sample"])
        H["d"]["sample_flow"] = {"rmse": 1.0}  # missing the other fields
        out = S.score_all(H, {"v": {"a": 4, "b": 3, "c": 2, "d": 1}})
        self.assertEqual(list(out), ["sample", "sample_flow"])
        self.assertEqual(out["sample_flow"]["datasets"], ["a", "b", "c"])
        self.assertIn("d", out["sample_flow"]["dropped_datasets"])
        self.assertLess(out["sample_flow"]["metrics"]["RMSE"]["conformance"], out["sample"]["metrics"]["RMSE"]["conformance"])


class CommandLine(unittest.TestCase):
    def test_end_to_end_writes_contract_shape_and_prints_table(self):
        H = hardness_from(RMSE={"a": 1, "b": 2, "c": 3, "d": 4}, ME={"a": 1, "b": 3, "c": 2, "d": 4})
        T = {"packed_rank": {"a": 4.0, "b": 2.0, "c": 3.0, "d": 1.0}, "sorted_vector": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, "tiny": {"a": 1.0}}
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "H.json").write_text(json.dumps(H))
            (root / "T.json").write_text(json.dumps(T))
            out = root / "nested" / "S.json"
            p = subprocess.run([sys.executable, str(TOOL), "--hardness", str(root / "H.json"), "--throughput", str(root / "T.json"),
                                "--output", str(out), "--print", "--ddof", "1"], capture_output=True, text=True, timeout=60)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("PLA-32" + DOT + "PLA-4096 (GRE)", p.stdout)
            self.assertIn("tiny: skipped", p.stdout)
            self.assertIn("not a reproduction", p.stdout)
            scores = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(list(scores), ["sample"])
            block = scores["sample"]
            self.assertEqual(block["ddof"], 1)
            self.assertEqual(block["variants"], ["packed_rank", "sorted_vector"])
            self.assertEqual(list(block["metrics"]), S.metric_names())
            for name, m in block["metrics"].items():
                self.assertEqual(m["dims"], name.split(DOT))
                for key in ("coverage", "conformance", "per_variant", "comparable_pairs", "incomparable_pairs"):
                    self.assertIn(key, m, name)
                self.assertEqual(m["comparable_pairs"] + m["incomparable_pairs"], 6)
                self.assertEqual(set(m["per_variant"]), {"packed_rank", "sorted_vector"})
                self.assertAlmostEqual(m["conformance"], sum(m["per_variant"].values()) / 2, places=12)
            self.assertEqual(block["metrics"]["RMSE"]["per_variant"]["sorted_vector"], -1.0)

    def test_rejects_unknown_option_and_missing_scope(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "H.json").write_text(json.dumps({"a": {"weird": {}}}))
            (root / "T.json").write_text(json.dumps({"v": {"a": 1}}))
            base = [sys.executable, str(TOOL), "--hardness", str(root / "H.json"), "--throughput", str(root / "T.json"), "--output", str(root / "S.json")]
            self.assertNotEqual(subprocess.run(base + ["--bogus"], capture_output=True, text=True).returncode, 0)
            p = subprocess.run(base, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("no scope", p.stderr)


if __name__ == "__main__":
    unittest.main()
