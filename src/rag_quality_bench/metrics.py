"""Transparent retrieval, citation, and uncertainty metrics."""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


class MetricError(ValueError):
    """Metric inputs are invalid or outside documented bounds."""


def reciprocal_rank(expected: set[str], ranked_sources: Sequence[str]) -> float:
    if not expected:
        return 1.0
    for index, source_id in enumerate(ranked_sources, start=1):
        if source_id in expected:
            return 1.0 / index
    return 0.0


def precision_at_k(expected: set[str], ranked_sources: Sequence[str], k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise MetricError("k must be a positive integer")
    selected = list(ranked_sources[:k])
    if not selected:
        return 1.0 if not expected else 0.0
    seen: set[str] = set()
    hits = 0
    for source_id in selected:
        if source_id in expected and source_id not in seen:
            hits += 1
            seen.add(source_id)
    return hits / len(selected)


def ndcg_at_k(expected: set[str], ranked_sources: Sequence[str], k: int) -> float:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise MetricError("k must be a positive integer")
    if not expected:
        return 1.0
    seen: set[str] = set()
    relevance: list[float] = []
    for source_id in ranked_sources[:k]:
        relevant = source_id in expected and source_id not in seen
        relevance.append(1.0 if relevant else 0.0)
        if relevant:
            seen.add(source_id)
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance))
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_hits))
    return 0.0 if idcg == 0 else dcg / idcg


def citation_scores(
    citations: set[str],
    expected_sources: set[str],
    valid_sources: set[str],
    retrieved_sources: set[str],
) -> tuple[float, float]:
    """Score citations against expected evidence that was valid and retrieved."""

    eligible = expected_sources & valid_sources & retrieved_sources
    if citations:
        precision = len(citations & eligible) / len(citations)
    else:
        precision = 1.0 if not expected_sources else 0.0
    recall = 1.0 if not expected_sources else len(citations & eligible) / len(expected_sources)
    return precision, recall


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Return a deterministic percentile bootstrap interval for a bounded mean."""

    samples = [float(value) for value in values]
    if not samples:
        return {"mean": 1.0, "lower": 1.0, "upper": 1.0, "iterations": 0, "seed": seed}
    if len(samples) > 10000:
        raise MetricError("bootstrap accepts at most 10000 observations")
    if any(not math.isfinite(value) for value in samples):
        raise MetricError("bootstrap observations must be finite")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 100 <= iterations <= 10000:
        raise MetricError("bootstrap iterations must be between 100 and 10000")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.5 <= confidence < 1:
        raise MetricError("bootstrap confidence must be between 0.5 and 1")
    rng = random.Random(seed)
    count = len(samples)
    means = sorted(sum(samples[rng.randrange(count)] for _ in range(count)) / count for _ in range(iterations))
    alpha = (1 - float(confidence)) / 2
    lower_index = max(0, min(iterations - 1, int(alpha * iterations)))
    upper_index = max(0, min(iterations - 1, int((1 - alpha) * iterations) - 1))
    return {
        "mean": round(sum(samples) / count, 6),
        "lower": round(means[lower_index], 6),
        "upper": round(means[upper_index], 6),
        "iterations": iterations,
        "seed": seed,
    }
