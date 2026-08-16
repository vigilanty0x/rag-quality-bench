from copy import deepcopy
import unittest
from unittest.mock import patch

from fixtures import StepClock, suite, suite_dict
from rag_quality_bench.engine import (
    BenchmarkEngine,
    chunk_document,
    compare_reports,
    retrieval_score,
    tokenize,
)
from rag_quality_bench.models import BenchmarkSuite, Document, content_sha256
from rag_quality_bench.retrieval import RetrievalError


class EngineTests(unittest.TestCase):
    def test_tokenize_is_case_insensitive_and_punctuation_free(self):
        self.assertEqual(tokenize("Hello, HELLO! 9"), ("hello", "hello"))

    def test_chunking_is_deterministic_and_overlapping(self):
        document = suite().documents[0]
        chunks = chunk_document(document, 8, 2)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_id, "handbook")
        self.assertTrue(chunks[1].chunk_id.startswith("handbook:1:"))
        self.assertEqual(chunks, chunk_document(document, 8, 2))

    def test_retrieval_score_is_transparent(self):
        chunk = chunk_document(suite().documents[0], 100, 0)[0]
        self.assertGreater(retrieval_score("launch city", chunk), 0)
        self.assertEqual(retrieval_score("submarine owner", chunk), 0)

    def test_happy_path_and_no_answer_pass(self):
        report = BenchmarkEngine(suite(), clock=StepClock(1_000_000)).run()
        self.assertEqual(report["metrics"]["pass_rate"], 1.0)
        self.assertTrue(all(record["passed"] for record in report["records"]))
        self.assertEqual(report["records"][0]["retrieval_latency_ms"], 1.0)

    def test_hash_mismatch_is_rejected_fail_closed(self):
        raw = suite_dict()
        raw["documents"][0]["content"] += " changed"
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertIn("hash_mismatch", report["inventory"]["documents"][0]["errors"])
        self.assertFalse(report["records"][0]["passed"])
        self.assertIn("retrieval_miss", report["records"][0]["failures"])

    def test_untrusted_document_is_not_indexed(self):
        raw = suite_dict()
        raw["documents"][0]["trust"] = "untrusted"
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertEqual(report["inventory"]["valid_document_count"], 0)
        self.assertIn("trust_untrusted", report["inventory"]["documents"][0]["errors"])

    def test_expired_document_is_not_indexed(self):
        raw = suite_dict()
        raw["documents"][0]["expires_at"] = "2026-01-01"
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertIn("expired", report["inventory"]["documents"][0]["errors"])

    def test_future_document_is_not_indexed(self):
        raw = suite_dict()
        raw["documents"][0]["observed_at"] = "2026-12-01"
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertIn("observed_in_future", report["inventory"]["documents"][0]["errors"])

    def test_duplicate_content_is_visible(self):
        raw = suite_dict()
        duplicate = dict(raw["documents"][0])
        duplicate["source_id"] = "handbook-copy"
        raw["documents"].append(duplicate)
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertEqual(report["inventory"]["duplicates"], [["handbook", "handbook-copy"]])

    def test_invalid_citation_is_preserved(self):
        raw = suite_dict()
        raw["questions"][0]["response"]["claims"][0]["citations"] = ["missing"]
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        first = report["records"][0]
        self.assertIn("invalid_citation", first["failures"])
        self.assertIn("missing_source:missing", first["claims"][0]["reasons"])

    def test_ungrounded_claim_is_preserved(self):
        raw = suite_dict()
        raw["questions"][0]["response"]["claims"][0]["text"] = "A lunar colony sells orange bicycles"
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertIn("ungrounded_claim", report["records"][0]["failures"])
        self.assertFalse(report["records"][0]["claims"][0]["supported"])

    def test_wrong_no_answer_behavior_fails(self):
        raw = suite_dict()
        raw["questions"][1]["response"] = {
            "answer": "It belongs to Ada.",
            "claims": [{"text": "It belongs to Ada", "citations": ["handbook"]}],
        }
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertIn("no_answer_mismatch", report["records"][1]["failures"])

    def test_latency_does_not_change_semantic_sha(self):
        slow = BenchmarkEngine(suite(), clock=StepClock(10_000_000)).run()
        fast = BenchmarkEngine(suite(), clock=StepClock(1_000)).run()
        self.assertNotEqual(slow["metrics"]["mean_retrieval_latency_ms"], fast["metrics"]["mean_retrieval_latency_ms"])
        self.assertEqual(slow["semantic_sha256"], fast["semantic_sha256"])

    def test_chunk_configuration_changes_index_sha(self):
        raw = suite_dict()
        first = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        raw["chunk_size"] = 8
        raw["chunk_overlap"] = 0
        second = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertNotEqual(first["index_sha256"], second["index_sha256"])

    def test_compare_reports_exposes_metric_and_question_changes(self):
        baseline = BenchmarkEngine(suite(), clock=StepClock(1)).run()
        raw = suite_dict()
        raw["questions"][0]["response"]["claims"][0]["text"] = "unsupported lunar colony statement"
        candidate = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        diff = compare_reports(baseline, candidate)
        self.assertLess(diff["metric_deltas"]["pass_rate"], 0)
        self.assertEqual(diff["changed_questions"][0]["question_id"], "launch-city")

    def test_fresh_coverage_counts_valid_documents(self):
        raw = suite_dict()
        content = "A blocked synthetic source."
        raw["documents"].append({
            "source_id": "blocked",
            "title": "Blocked",
            "source_url": "https://example.invalid/blocked",
            "license": "CC0-1.0",
            "observed_at": "2026-08-14",
            "trust": "blocked",
            "content": content,
            "sha256": content_sha256(content),
        })
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        self.assertEqual(report["metrics"]["fresh_coverage"], 0.5)

    def test_bm25_and_hybrid_reports_include_ranking_metrics(self):
        for strategy in ("bm25", "tfidf", "hybrid"):
            raw = suite_dict()
            raw["retrieval_strategy"] = strategy
            report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
            first = report["records"][0]
            self.assertEqual(first["retrieval_strategy"], strategy)
            self.assertEqual(first["reciprocal_rank"], 1.0)
            self.assertEqual(first["ndcg_at_k"], 1.0)
            self.assertIn("citation_precision", first)
            self.assertIn("pass_rate_ci95", report["metrics"])

    def test_citation_metrics_require_retrieved_evidence(self):
        raw = suite_dict()
        raw["questions"][0]["text"] = "xylophone quasar"
        report = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1)).run()
        record = report["records"][0]
        self.assertEqual(record["retrieved"], [])
        self.assertFalse(record["citations_valid"])
        self.assertEqual(record["citation_precision"], 0.0)
        self.assertEqual(record["citation_recall"], 0.0)

    def test_chunk_limit_is_rejected_before_chunk_materialization(self):
        raw = suite_dict()
        content = ("aa " * 66_666).strip()
        raw["documents"][0]["content"] = content
        raw["documents"][0]["sha256"] = content_sha256(content)
        duplicate = deepcopy(raw["documents"][0])
        duplicate["source_id"] = "handbook-copy"
        raw["documents"].append(duplicate)
        raw["chunk_size"] = 8
        raw["chunk_overlap"] = 7
        benchmark = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1))
        with patch("rag_quality_bench.engine.chunk_document") as materialize:
            with self.assertRaisesRegex(RetrievalError, "100000"):
                benchmark.run()
        materialize.assert_not_called()

    def test_new_reports_use_explicit_report_schema_two(self):
        report = BenchmarkEngine(suite(), clock=StepClock(1)).run()
        self.assertEqual(report["schema_version"], "2.0")

    def test_index_manifest_detects_strategy_and_is_content_redacted(self):
        raw = suite_dict()
        engine = BenchmarkEngine(BenchmarkSuite.from_dict(raw), clock=StepClock(1))
        manifest = engine.index_manifest()
        self.assertTrue(engine.verify_index_manifest(manifest))
        self.assertNotIn("launch city is Geneva", str(manifest))
        manifest["chunks"][0]["token_count"] += 1
        self.assertFalse(engine.verify_index_manifest(manifest))
