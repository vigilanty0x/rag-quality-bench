import unittest

from rag_quality_bench.probes import functional_probe, liveness_probe, readiness_probe


class ProbeTests(unittest.TestCase):
    def test_liveness_exposes_version(self):
        result = liveness_probe()
        self.assertTrue(result["ok"])
        self.assertRegex(result["version"], r"^\d+\.\d+\.\d+$")

    def test_readiness_has_no_runtime_dependencies(self):
        result = readiness_probe()
        self.assertTrue(result["ok"])
        self.assertEqual(result["runtime_dependencies"], [])
        self.assertEqual(result["report_schema_versions"], ["1.0", "2.0"])

    def test_functional_probe_requires_counter_failure(self):
        result = functional_probe()
        self.assertTrue(result["ok"])
        self.assertTrue(result["control_passed"])
        self.assertTrue(result["counter_example_failed"])
        self.assertIn("invalid_citation", result["counter_failures"])
