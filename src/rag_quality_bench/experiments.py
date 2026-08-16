"""Bounded, reproducible retrieval configuration sweeps."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .engine import BenchmarkEngine
from .models import BenchmarkSuite, RETRIEVAL_STRATEGIES, canonical_sha256

MAX_SWEEP_RUNS = 32
MAX_SWEEP_DIMENSION = 32


class ExperimentError(ValueError):
    """An experiment definition is invalid or unreasonably large."""


def _bounded_unique(values: list[Any], label: str) -> None:
    if not isinstance(values, list) or not values:
        raise ExperimentError(f"{label} must be a non-empty unique array")
    if len(values) > MAX_SWEEP_DIMENSION:
        raise ExperimentError(f"{label} dimension may contain at most {MAX_SWEEP_DIMENSION} values")
    try:
        unique_count = len(set(values))
    except TypeError as exc:
        raise ExperimentError(f"{label} values must be scalar") from exc
    if len(values) != unique_count:
        raise ExperimentError(f"{label} must be a non-empty unique array")


def run_sweep(
    suite: BenchmarkSuite,
    *,
    strategies: list[str],
    chunk_sizes: list[int],
    overlaps: list[int],
    clock_factory: Callable[[], Callable[[], int]] | None = None,
) -> dict[str, Any]:
    _bounded_unique(strategies, "strategies")
    _bounded_unique(chunk_sizes, "chunk_sizes")
    _bounded_unique(overlaps, "overlaps")
    if any(not isinstance(strategy, str) or strategy not in RETRIEVAL_STRATEGIES for strategy in strategies):
        raise ExperimentError(f"strategies must use {sorted(RETRIEVAL_STRATEGIES)}")
    if any(isinstance(size, bool) or not isinstance(size, int) or not 8 <= size <= 500 for size in chunk_sizes):
        raise ExperimentError("chunk_sizes must contain integers from 8 to 500")
    if any(isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0 for overlap in overlaps):
        raise ExperimentError("overlaps must contain non-negative integers")
    configurations: list[tuple[str, int, int]] = []
    for strategy in strategies:
        for size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= size:
                    continue
                if len(configurations) >= MAX_SWEEP_RUNS:
                    raise ExperimentError(f"a sweep may contain at most {MAX_SWEEP_RUNS} runs")
                configurations.append((strategy, size, overlap))
    if not configurations:
        raise ExperimentError("no valid configuration remains after enforcing overlap < chunk_size")

    runs: list[dict[str, Any]] = []
    for strategy, size, overlap in configurations:
        candidate = replace(
            suite,
            retrieval_strategy=strategy,
            chunk_size=size,
            chunk_overlap=overlap,
        )
        clock = None if clock_factory is None else clock_factory()
        report = BenchmarkEngine(candidate, clock=clock).run()
        configuration = {
            "strategy": strategy,
            "chunk_size": size,
            "chunk_overlap": overlap,
            "retrieval_k": suite.retrieval_k,
            "hybrid_weight": suite.hybrid_weight,
        }
        run_id = canonical_sha256({"suite": suite.suite_id, "configuration": configuration})[:16]
        metrics = report["metrics"]
        quality_score = (
            0.4 * metrics["pass_rate"]
            + 0.2 * metrics["recall_at_k"]
            + 0.15 * metrics["ndcg_at_k"]
            + 0.15 * metrics["groundedness"]
            + 0.1 * metrics["no_answer_accuracy"]
        )
        runs.append(
            {
                "run_id": run_id,
                "configuration": configuration,
                "quality_score": round(quality_score, 6),
                "metrics": metrics,
                "semantic_sha256": report["semantic_sha256"],
                "index_sha256": report["index_sha256"],
            }
        )
    runs.sort(key=lambda row: row["run_id"])
    best = sorted(
        runs,
        key=lambda row: (
            -row["quality_score"],
            -row["metrics"]["pass_rate"],
            -row["metrics"]["recall_at_k"],
            row["run_id"],
        ),
    )[0]
    semantic_runs = [
        {
            **row,
            "metrics": {
                key: value
                for key, value in row["metrics"].items()
                if key != "mean_retrieval_latency_ms"
            },
        }
        for row in runs
    ]
    semantic = {
        "experiment_version": "1.0",
        "suite_id": suite.suite_id,
        "suite_sha256": suite.digest,
        "best_run_id": best["run_id"],
        "runs": semantic_runs,
    }
    return {
        "experiment_version": "1.0",
        "status": "verified",
        "suite_id": suite.suite_id,
        "suite_sha256": suite.digest,
        "best_run_id": best["run_id"],
        "runs": runs,
        "semantic_sha256": canonical_sha256(semantic),
    }


def write_sweep(path: str | Path, sweep: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists() and target.is_dir():
        raise ExperimentError("sweep output path cannot be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(sweep, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
