import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile

from fixtures import suite
from rag_quality_bench.engine import chunk_document
from rag_quality_bench.models import canonical_sha256
from rag_quality_bench.retrieval import RetrievalIndex, RetrievalError, load_manifest, verify_manifest


class RetrievalTests(unittest.TestCase):
    def test_every_local_strategy_ranks_the_relevant_chunk(self):
        chunks = list(chunk_document(suite().documents[0], 100, 0))
        for strategy in ("overlap", "bm25", "tfidf", "hybrid"):
            index = RetrievalIndex(chunks, strategy=strategy, hybrid_weight=0.6)
            ranked = index.rank("Where is the launch city?", limit=3)
            self.assertEqual(ranked[0].chunk.source_id, "handbook")
            self.assertGreater(ranked[0].score, 0)

    def test_invalid_strategy_and_limits_fail_closed(self):
        chunks = list(chunk_document(suite().documents[0], 100, 0))
        with self.assertRaises(RetrievalError):
            RetrievalIndex(chunks, strategy="remote")
        with self.assertRaises(RetrievalError):
            RetrievalIndex(chunks).rank("query", limit=0)

    def test_extreme_hybrid_weight_is_rejected_without_overflow(self):
        chunks = list(chunk_document(suite().documents[0], 100, 0))
        huge = 10**1000
        with self.assertRaisesRegex(RetrievalError, "hybrid_weight"):
            RetrievalIndex(chunks, strategy="hybrid", hybrid_weight=huge)

        manifest = RetrievalIndex(chunks).manifest(
            suite_sha256=suite().digest,
            chunk_size=100,
            chunk_overlap=0,
        )
        manifest["hybrid_weight"] = huge
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256")
        manifest["manifest_sha256"] = canonical_sha256(unsigned)
        self.assertFalse(verify_manifest(manifest))

    def test_index_stops_consuming_an_oversized_iterable_at_the_bound(self):
        chunks = list(chunk_document(suite().documents[0], 8, 2))
        consumed = []

        def source():
            for index in range(100):
                consumed.append(index)
                chunk = chunks[index % len(chunks)]
                yield type(chunk)(
                    f"{chunk.chunk_id}:{index}",
                    chunk.source_id,
                    index,
                    chunk.text,
                    chunk.document_sha256,
                )

        with patch("rag_quality_bench.retrieval.MAX_INDEX_CHUNKS", 2):
            with self.assertRaisesRegex(RetrievalError, "1..2"):
                RetrievalIndex(source())
        self.assertEqual(consumed, [0, 1, 2])

    def test_hybrid_ranking_is_deterministic(self):
        chunks = list(chunk_document(suite().documents[0], 8, 2))
        index = RetrievalIndex(chunks, strategy="hybrid", hybrid_weight=0.4)
        first = [(row.chunk.chunk_id, row.score) for row in index.rank("launch city", limit=5)]
        second = [(row.chunk.chunk_id, row.score) for row in index.rank("launch city", limit=5)]
        self.assertEqual(first, second)

    def test_self_rehashed_malformed_manifest_is_rejected(self):
        chunks = list(chunk_document(suite().documents[0], 8, 2))
        manifest = RetrievalIndex(chunks, strategy="bm25").manifest(
            suite_sha256=suite().digest,
            chunk_size=8,
            chunk_overlap=2,
        )
        manifest["chunks"][0]["token_count"] = -1
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256")
        manifest["manifest_sha256"] = canonical_sha256(unsigned)
        self.assertFalse(verify_manifest(manifest))

    def test_manifest_loader_rejects_duplicate_members(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"index_version":"1.0","index_version":"2.0"}', encoding="utf-8")
            with self.assertRaisesRegex(RetrievalError, "duplicate JSON key"):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
