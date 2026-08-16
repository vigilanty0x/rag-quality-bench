from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from fixtures import StepClock, suite
from rag_quality_bench.engine import BenchmarkEngine
from rag_quality_bench.models import ContractError, canonical_sha256
from rag_quality_bench.reporting import (
    export_report,
    load_report,
    render_html,
    render_markdown,
    semantic_payload,
    verify_report,
    write_report,
)
from rag_quality_bench.retrieval import verify_manifest


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.report = BenchmarkEngine(suite(), clock=StepClock(1_000)).run()

    def test_report_verifies(self):
        self.assertTrue(verify_report(self.report))

    def test_legacy_schema_one_report_remains_verifiable(self):
        legacy = deepcopy(self.report)
        legacy["schema_version"] = "1.0"
        legacy.pop("index_manifest")
        released_payload = {
            "suite_sha256": legacy["suite_sha256"],
            "index_sha256": legacy["index_sha256"],
            "inventory": legacy["inventory"],
            "metrics": {
                key: value for key, value in legacy["metrics"].items() if "latency" not in key
            },
            "records": [
                {
                    key: value
                    for key, value in record.items()
                    if key != "retrieval_latency_ms"
                }
                for record in legacy["records"]
            ],
        }
        legacy["semantic_sha256"] = canonical_sha256(released_payload)
        self.assertTrue(verify_report(legacy))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(legacy), encoding="utf-8")
            self.assertEqual(load_report(path)["schema_version"], "1.0")

    def test_atomic_write_and_reopen(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            write_report(path, self.report)
            self.assertEqual(load_report(path)["semantic_sha256"], self.report["semantic_sha256"])

    def test_tampered_report_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, self.report)
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["records"][0]["passed"] = False
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "verification failed"):
                load_report(path)

    def test_metadata_or_unknown_field_tampering_fails(self):
        changed = dict(self.report)
        changed["suite_id"] = "forged-suite"
        self.assertFalse(verify_report(changed))
        changed = dict(self.report)
        changed["unhashed_note"] = "injected"
        self.assertFalse(verify_report(changed))
        changed = dict(self.report)
        changed["metrics"] = {**self.report["metrics"], "forged_latency_claim": "trusted"}
        self.assertFalse(verify_report(changed))

    def test_rehashed_invalid_nested_manifest_and_cross_links_fail(self):
        changed = deepcopy(self.report)
        changed["index_manifest"]["chunks"][0]["token_count"] = -1
        changed["index_sha256"] = "0" * 64
        changed["semantic_sha256"] = canonical_sha256(semantic_payload(changed))
        self.assertFalse(verify_manifest(changed["index_manifest"]))
        self.assertFalse(verify_report(changed))

    def test_rehashed_logically_inconsistent_metric_fails(self):
        changed = deepcopy(self.report)
        changed["metrics"]["pass_rate"] = 0.25
        changed["semantic_sha256"] = canonical_sha256(semantic_payload(changed))
        self.assertFalse(verify_report(changed))

    def test_duplicate_report_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            payload = json.dumps(self.report).replace('"suite_id":', '"suite_id":"shadow","suite_id":', 1)
            path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
                load_report(path)

    def test_invalid_json_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(ContractError):
                load_report(path)

    def test_missing_report_fails(self):
        with self.assertRaisesRegex(ContractError, "does not exist"):
            load_report("/tmp/definitely-not-a-rag-quality-report.json")

    def test_directory_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ContractError, "directory"):
                write_report(directory, self.report)

    def test_markdown_and_html_exports_are_safe_and_useful(self):
        markdown = render_markdown(self.report)
        html = render_html(self.report)
        self.assertIn("RAG Quality Bench", markdown)
        self.assertIn("<!doctype html>", html.lower())
        self.assertIn("Pass rate", html)
        with tempfile.TemporaryDirectory() as directory:
            md_path = Path(directory) / "report.md"
            html_path = Path(directory) / "report.html"
            export_report(md_path, self.report, format="markdown")
            export_report(html_path, self.report, format="html")
            self.assertTrue(md_path.is_file())
            self.assertTrue(html_path.is_file())

    def test_markdown_cell_neutralizes_raw_html(self):
        from rag_quality_bench.reporting import _markdown_cell

        payload = "<img src=x onerror=alert(1)>"
        rendered = _markdown_cell(payload)
        self.assertNotIn(payload, rendered)
        self.assertIn("&lt;img", rendered)

    def test_invalid_report_does_not_replace_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            write_report(path, self.report)
            original = path.read_bytes()
            changed = deepcopy(self.report)
            changed["records"][0]["passed"] = False
            with self.assertRaisesRegex(ContractError, "invalid report"):
                write_report(path, changed)
            self.assertEqual(path.read_bytes(), original)
