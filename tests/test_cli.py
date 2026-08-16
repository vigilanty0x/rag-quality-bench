from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from rag_quality_bench.cli import main


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "suite.json"


class CliTests(unittest.TestCase):
    def invoke(self, argv):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_validate(self):
        code, stdout, _ = self.invoke(["validate", "--suite", str(EXAMPLE)])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["valid"])

    def test_inventory(self):
        code, stdout, _ = self.invoke(["inventory", "--suite", str(EXAMPLE)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout)["valid_document_count"], 1)

    def test_run_writes_verified_report_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            code, stdout, _ = self.invoke(["run", "--suite", str(EXAMPLE), "--output", str(path)])
            self.assertEqual(code, 0)
            report = json.loads(stdout)
            self.assertTrue(path.is_file())
            self.assertLess(report["metrics"]["pass_rate"], 1)
            self.assertTrue(any(row["failures"] for row in report["records"]))
            verify_code, _, _ = self.invoke(["verify", "--report", str(path)])
            self.assertEqual(verify_code, 0)

    def test_minimum_pass_rate_can_fail_gate(self):
        code, _, _ = self.invoke(["run", "--suite", str(EXAMPLE), "--minimum-pass-rate", "1"])
        self.assertEqual(code, 1)

    def test_invalid_threshold_is_bounded_error(self):
        code, _, stderr = self.invoke(["run", "--suite", str(EXAMPLE), "--minimum-pass-rate", "2"])
        self.assertEqual(code, 2)
        self.assertIn("between 0 and 1", stderr)

    def test_functional_probe(self):
        code, stdout, _ = self.invoke(["probe", "--level", "functional"])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(stdout)["counter_example_failed"])

    def test_compare_verified_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            self.invoke(["run", "--suite", str(EXAMPLE), "--output", str(first)])
            self.invoke(["run", "--suite", str(EXAMPLE), "--output", str(second)])
            code, stdout, _ = self.invoke(["compare", "--baseline", str(first), "--candidate", str(second)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout)["metric_deltas"]["pass_rate"], 0)

    def test_missing_suite_is_bounded_error(self):
        code, _, stderr = self.invoke(["validate", "--suite", "/tmp/no-rag-suite.json"])
        self.assertEqual(code, 2)
        self.assertIn("does not exist", stderr)

    def test_suite_size_is_rejected_before_an_unbounded_read(self):
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory) / "oversized.json"
            oversized.write_bytes(b" " * 5_000_001)
            code, _, stderr = self.invoke(["validate", "--suite", str(oversized)])
            self.assertEqual(code, 2)
            self.assertIn("exceeds 5 MB", stderr)

    def test_extreme_numeric_input_has_a_bounded_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
            raw["hybrid_weight"] = 10**400
            path = Path(directory) / "extreme.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            code, _, stderr = self.invoke(["validate", "--suite", str(path)])
            self.assertEqual(code, 2)
            self.assertIn("hybrid_weight", stderr)
            self.assertNotIn("Traceback", stderr)

    def test_demo_is_reproducible_logically(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "one.json"
            second = Path(directory) / "two.json"
            code1, out1, _ = self.invoke(["demo", "--output", str(first)])
            code2, out2, _ = self.invoke(["demo", "--output", str(second)])
            self.assertEqual((code1, code2), (0, 0))
            self.assertEqual(json.loads(out1)["semantic_sha256"], json.loads(out2)["semantic_sha256"])

    def test_index_sweep_and_export_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "index.json"
            code, stdout, _ = self.invoke(["index", "--suite", str(EXAMPLE), "--strategy", "bm25", "--output", str(manifest)])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads(stdout)["valid"])
            verify_code, _, _ = self.invoke(["verify-index", "--manifest", str(manifest)])
            self.assertEqual(verify_code, 0)

            sweep = root / "sweep.json"
            code, stdout, _ = self.invoke(["sweep", "--suite", str(EXAMPLE), "--strategies", "overlap,bm25", "--chunk-sizes", "12,20", "--overlaps", "0,2", "--output", str(sweep)])
            self.assertEqual(code, 0)
            self.assertEqual(len(json.loads(stdout)["runs"]), 8)

            report = root / "report.json"
            self.invoke(["run", "--suite", str(EXAMPLE), "--output", str(report)])
            output = root / "report.html"
            code, _, _ = self.invoke(["export", "--report", str(report), "--format", "html", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertIn("<!doctype html>", output.read_text(encoding="utf-8").lower())
