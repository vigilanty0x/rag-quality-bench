import unittest

from fixtures import StepClock, suite
from rag_quality_bench.experiments import ExperimentError, run_sweep


class ExperimentTests(unittest.TestCase):
    def test_sweep_compares_real_strategies_and_selects_a_best_run(self):
        result = run_sweep(
            suite(),
            strategies=["overlap", "bm25"],
            chunk_sizes=[8, 12],
            overlaps=[0, 2],
            clock_factory=lambda: StepClock(1),
        )
        self.assertEqual(len(result["runs"]), 8)
        self.assertIn(result["best_run_id"], {row["run_id"] for row in result["runs"]})
        self.assertEqual(result["status"], "verified")

    def test_sweep_bounds_combinatorial_explosion(self):
        with self.assertRaisesRegex(ExperimentError, "at most"):
            run_sweep(
                suite(),
                strategies=["overlap", "bm25", "tfidf", "hybrid"],
                chunk_sizes=list(range(8, 20)),
                overlaps=[0, 1],
            )

    def test_sweep_bounds_each_dimension_before_cartesian_expansion(self):
        with self.assertRaisesRegex(ExperimentError, "dimension"):
            run_sweep(
                suite(),
                strategies=["overlap"],
                chunk_sizes=[8],
                overlaps=list(range(501, 534)),
            )

    def test_sweep_semantic_result_ignores_input_order_and_latency(self):
        first = run_sweep(
            suite(),
            strategies=["overlap", "bm25"],
            chunk_sizes=[8, 12],
            overlaps=[0, 2],
            clock_factory=lambda: StepClock(1),
        )
        second = run_sweep(
            suite(),
            strategies=["bm25", "overlap"],
            chunk_sizes=[12, 8],
            overlaps=[2, 0],
            clock_factory=lambda: StepClock(1000),
        )
        self.assertEqual(first["semantic_sha256"], second["semantic_sha256"])
        self.assertEqual(
            [row["run_id"] for row in first["runs"]],
            [row["run_id"] for row in second["runs"]],
        )


if __name__ == "__main__":
    unittest.main()
