import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PackagingContractTests(unittest.TestCase):
    def test_release_toolchain_matches_build_system_exactly(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = {
            line.strip()
            for line in (ROOT / "requirements-build.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertEqual(
            set(project["build-system"]["requires"]),
            requirements - {"pip==25.2"},
        )
        self.assertTrue(all("==" in requirement for requirement in requirements))

    def test_sdist_manifest_contains_release_evidence_and_fixtures(self):
        manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
        for expected in (
            "include requirements-build.txt",
            "recursive-include examples *.json",
            "recursive-include scripts *.py",
            "recursive-include tests *.py",
        ):
            self.assertIn(expected, manifest)

    def test_bundled_demo_matches_the_public_example_semantically(self):
        public = json.loads((ROOT / "examples" / "suite.json").read_text(encoding="utf-8"))
        bundled = json.loads(
            (ROOT / "src" / "rag_quality_bench" / "data" / "suite.json").read_text(encoding="utf-8")
        )
        self.assertEqual(public, bundled)


if __name__ == "__main__":
    unittest.main()
