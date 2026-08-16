import unittest

from rag_quality_bench.metrics import (
    bootstrap_mean_ci,
    citation_scores,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
)


class MetricTests(unittest.TestCase):
    def test_rank_metrics_use_expected_sources(self):
        ranked = ["noise", "target", "target"]
        self.assertEqual(reciprocal_rank({"target"}, ranked), 0.5)
        self.assertGreater(ndcg_at_k({"target"}, ranked, 3), 0)
        self.assertEqual(precision_at_k({"target"}, ranked, 2), 0.5)

    def test_citation_precision_and_recall(self):
        precision, recall = citation_scores(
            {"a", "bad"},
            {"a", "b"},
            {"a", "b", "c"},
            {"a", "c"},
        )
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 0.5)

    def test_citation_scores_do_not_credit_unretrieved_sources(self):
        precision, recall = citation_scores({"a"}, {"a"}, {"a"}, set())
        self.assertEqual((precision, recall), (0.0, 0.0))

    def test_bootstrap_interval_is_seeded_and_bounded(self):
        first = bootstrap_mean_ci([0.0, 1.0, 1.0], iterations=200, seed=7)
        second = bootstrap_mean_ci([0.0, 1.0, 1.0], iterations=200, seed=7)
        self.assertEqual(first, second)
        self.assertLessEqual(first["lower"], first["mean"])
        self.assertLessEqual(first["mean"], first["upper"])


if __name__ == "__main__":
    unittest.main()
