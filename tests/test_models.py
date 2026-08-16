import json
import unittest

from fixtures import suite_dict
from rag_quality_bench.models import BenchmarkSuite, ContractError, content_sha256


class SuiteContractTests(unittest.TestCase):
    def test_valid_suite_round_trip_and_digest(self):
        first = BenchmarkSuite.from_dict(suite_dict())
        second = BenchmarkSuite.from_json(json.dumps(first.to_dict()))
        self.assertEqual(first, second)
        self.assertEqual(first.digest, second.digest)

    def test_digest_ignores_mapping_order(self):
        raw = suite_dict()
        reversed_raw = dict(reversed(list(raw.items())))
        self.assertEqual(BenchmarkSuite.from_dict(raw).digest, BenchmarkSuite.from_dict(reversed_raw).digest)

    def test_default_retrieval_preserves_released_schema_one_digest(self):
        raw = suite_dict()
        omitted = BenchmarkSuite.from_dict(raw)
        raw["retrieval_strategy"] = "overlap"
        raw["hybrid_weight"] = 0.5
        explicit = BenchmarkSuite.from_dict(raw)
        self.assertEqual(omitted.digest, explicit.digest)
        self.assertEqual(
            omitted.digest,
            "e83fd2e9a6675c3104f9608913efb86eb04c917915cafa4b5a97c69568633bb9",
        )

    def test_unknown_suite_field_fails(self):
        raw = suite_dict()
        raw["secret_mode"] = True
        with self.assertRaisesRegex(ContractError, "unknown"):
            BenchmarkSuite.from_dict(raw)

    def test_schema_version_is_exact(self):
        raw = suite_dict()
        raw["schema_version"] = "2.0"
        with self.assertRaises(ContractError):
            BenchmarkSuite.from_dict(raw)

    def test_semver_is_required(self):
        raw = suite_dict()
        raw["version"] = "latest"
        with self.assertRaisesRegex(ContractError, "semantic"):
            BenchmarkSuite.from_dict(raw)

    def test_chunk_bounds_are_enforced(self):
        for value in (7, 501, "12"):
            raw = suite_dict()
            raw["chunk_size"] = value
            with self.assertRaises(ContractError):
                BenchmarkSuite.from_dict(raw)

    def test_overlap_must_be_smaller_than_chunk(self):
        raw = suite_dict()
        raw["chunk_overlap"] = raw["chunk_size"]
        with self.assertRaises(ContractError):
            BenchmarkSuite.from_dict(raw)

    def test_retrieval_k_is_bounded(self):
        raw = suite_dict()
        raw["retrieval_k"] = 0
        with self.assertRaises(ContractError):
            BenchmarkSuite.from_dict(raw)

    def test_duplicate_source_ids_fail(self):
        raw = suite_dict()
        raw["documents"].append(dict(raw["documents"][0]))
        with self.assertRaisesRegex(ContractError, "unique"):
            BenchmarkSuite.from_dict(raw)

    def test_duplicate_question_ids_fail(self):
        raw = suite_dict()
        raw["questions"].append(dict(raw["questions"][0]))
        with self.assertRaisesRegex(ContractError, "unique"):
            BenchmarkSuite.from_dict(raw)

    def test_unknown_expected_source_fails(self):
        raw = suite_dict()
        raw["questions"][0]["expected_source_ids"] = ["missing"]
        with self.assertRaisesRegex(ContractError, "unknown expected"):
            BenchmarkSuite.from_dict(raw)

    def test_no_answer_cannot_expect_source(self):
        raw = suite_dict()
        raw["questions"][0]["no_answer"] = True
        with self.assertRaises(ContractError):
            BenchmarkSuite.from_dict(raw)

    def test_answerable_question_requires_source(self):
        raw = suite_dict()
        raw["questions"][0]["expected_source_ids"] = []
        with self.assertRaises(ContractError):
            BenchmarkSuite.from_dict(raw)

    def test_answer_requires_claim(self):
        raw = suite_dict()
        raw["questions"][0]["response"]["claims"] = []
        with self.assertRaisesRegex(ContractError, "must contain claims"):
            BenchmarkSuite.from_dict(raw)

    def test_abstention_cannot_have_claim(self):
        raw = suite_dict()
        raw["questions"][1]["response"]["claims"] = [{"text": "x", "citations": ["handbook"]}]
        with self.assertRaises(ContractError):
            BenchmarkSuite.from_dict(raw)

    def test_duplicate_claim_citations_fail(self):
        raw = suite_dict()
        raw["questions"][0]["response"]["claims"][0]["citations"] = ["handbook", "handbook"]
        with self.assertRaisesRegex(ContractError, "duplicates"):
            BenchmarkSuite.from_dict(raw)

    def test_duplicate_expected_source_ids_fail(self):
        raw = suite_dict()
        raw["questions"][0]["expected_source_ids"] = ["handbook", "handbook"]
        with self.assertRaisesRegex(ContractError, "expected_source_ids contains duplicates"):
            BenchmarkSuite.from_dict(raw)

    def test_invalid_document_hash_shape_fails_contract(self):
        raw = suite_dict()
        raw["documents"][0]["sha256"] = "bad"
        with self.assertRaisesRegex(ContractError, "SHA-256"):
            BenchmarkSuite.from_dict(raw)

    def test_content_hash_is_utf8_sha256(self):
        self.assertEqual(content_sha256("hello"), "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")

    def test_invalid_json_has_bounded_error(self):
        with self.assertRaisesRegex(ContractError, "invalid JSON"):
            BenchmarkSuite.from_json("{")

    def test_duplicate_json_keys_are_rejected(self):
        raw = json.dumps(suite_dict()).replace('"suite_id":', '"suite_id":"shadow","suite_id":', 1)
        with self.assertRaisesRegex(ContractError, "duplicate JSON key"):
            BenchmarkSuite.from_json(raw)

    def test_retrieval_strategy_and_hybrid_weight_are_bounded(self):
        raw = suite_dict()
        raw["retrieval_strategy"] = "bm25"
        raw["hybrid_weight"] = 0.7
        parsed = BenchmarkSuite.from_dict(raw)
        self.assertEqual(parsed.retrieval_strategy, "bm25")
        self.assertEqual(parsed.hybrid_weight, 0.7)
        raw["retrieval_strategy"] = "opaque-remote-model"
        with self.assertRaisesRegex(ContractError, "retrieval_strategy"):
            BenchmarkSuite.from_dict(raw)
        raw = suite_dict()
        raw["hybrid_weight"] = 2
        with self.assertRaisesRegex(ContractError, "hybrid_weight"):
            BenchmarkSuite.from_dict(raw)

    def test_extreme_hybrid_weight_is_a_contract_error(self):
        raw = suite_dict()
        raw["hybrid_weight"] = 10**400
        with self.assertRaisesRegex(ContractError, "hybrid_weight"):
            BenchmarkSuite.from_dict(raw)

    def test_extreme_json_integer_is_a_contract_error(self):
        payload = json.dumps(suite_dict()).replace('"chunk_size": 12', '"chunk_size": ' + "9" * 5000)
        with self.assertRaisesRegex(ContractError, "invalid JSON"):
            BenchmarkSuite.from_json(payload)
