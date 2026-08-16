"""Liveness, readiness, and functional counter-proof probes."""

from __future__ import annotations

from typing import Any

from . import __version__
from .engine import BenchmarkEngine
from .models import BenchmarkSuite, content_sha256


def liveness_probe() -> dict[str, Any]:
    return {"ok": True, "probe": "liveness", "version": __version__}


def readiness_probe() -> dict[str, Any]:
    return {
        "ok": True,
        "probe": "readiness",
        "runtime_dependencies": [],
        "schema_versions": ["1.0"],
        "report_schema_versions": ["1.0", "2.0"],
        "judges": ["retrieval", "citation", "groundedness", "no_answer"],
    }


def _probe_suite(citation: str) -> BenchmarkSuite:
    content = "Public release notes say the launch date is May 9 and the color is blue."
    return BenchmarkSuite.from_dict({
        "schema_version": "1.0",
        "suite_id": "functional-probe",
        "version": "1.0.0",
        "evaluation_date": "2026-05-10",
        "chunk_size": 20,
        "chunk_overlap": 2,
        "retrieval_k": 2,
        "documents": [{
            "source_id": "release-notes",
            "title": "Release notes",
            "source_url": "https://example.invalid/release",
            "license": "CC0-1.0",
            "observed_at": "2026-05-09",
            "trust": "trusted",
            "content": content,
            "sha256": content_sha256(content),
        }],
        "questions": [{
            "question_id": "launch-date",
            "text": "What is the launch date?",
            "expected_source_ids": ["release-notes"],
            "no_answer": False,
            "adversarial": False,
            "response": {
                "answer": "The launch date is May 9.",
                "claims": [{"text": "launch date is May 9", "citations": [citation]}],
            },
        }],
    })


def functional_probe() -> dict[str, Any]:
    control = BenchmarkEngine(_probe_suite("release-notes"), clock=iter([0, 1_000_000]).__next__).run()
    counter = BenchmarkEngine(_probe_suite("missing-source"), clock=iter([0, 1_000_000]).__next__).run()
    control_passed = control["records"][0]["passed"]
    counter_failed = not counter["records"][0]["passed"] and "invalid_citation" in counter["records"][0]["failures"]
    return {
        "ok": control_passed and counter_failed,
        "probe": "functional",
        "control_passed": control_passed,
        "counter_example_failed": counter_failed,
        "counter_failures": counter["records"][0]["failures"],
    }
