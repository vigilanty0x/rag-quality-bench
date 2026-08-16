"""Atomic report output and semantic verification."""

from __future__ import annotations

import hashlib
import json
from html import escape
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .metrics import bootstrap_mean_ci, citation_scores, ndcg_at_k, precision_at_k, reciprocal_rank
from .models import (
    ID_RE,
    MAX_DOCUMENTS,
    MAX_QUESTIONS,
    RETRIEVAL_STRATEGIES,
    SEMVER_RE,
    SHA_RE,
    ContractError,
    canonical_sha256,
)
from .retrieval import verify_manifest

MAX_REPORT_BYTES = 64 * 1024 * 1024
REPORT_FIELDS = {
    "schema_version", "suite_id", "suite_version", "suite_sha256", "index_sha256",
    "index_manifest", "inventory", "metrics", "records", "semantic_sha256",
}
LEGACY_REPORT_FIELDS = REPORT_FIELDS - {"index_manifest"}
INVENTORY_FIELDS = {
    "suite_sha256", "document_count", "valid_document_count", "fresh_coverage",
    "duplicates", "documents",
}
INVENTORY_DOCUMENT_FIELDS = {"source_id", "sha256", "valid", "errors"}
METRIC_FIELDS = {
    "recall_at_k", "precision_at_k", "mean_reciprocal_rank", "ndcg_at_k",
    "groundedness", "citation_precision", "citation_recall", "no_answer_accuracy",
    "fresh_coverage", "pass_rate", "adversarial_pass_rate", "pass_rate_ci95",
    "mean_retrieval_latency_ms", "retrieval_k", "retrieval_strategy",
}
RECORD_FIELDS = {
    "question_id", "adversarial", "expected_source_ids", "retrieved",
    "retrieval_strategy", "retrieval_latency_ms", "recall_at_k", "precision_at_k",
    "reciprocal_rank", "ndcg_at_k", "no_answer_correct", "citations_valid",
    "citation_precision", "citation_recall", "citation_errors", "claims", "grounded",
    "passed", "failures",
}
RETRIEVED_FIELDS = {
    "score", "lexical_score", "vector_score", "chunk_id", "source_id", "ordinal",
    "text", "document_sha256",
}
CLAIM_FIELDS = {"text", "citations", "supported", "token_coverage", "reasons"}
FAILURE_CODES = {
    "retrieval_miss", "invalid_citation", "no_answer_mismatch", "ungrounded_claim",
}
DOCUMENT_ERROR_CODES = {
    "hash_mismatch", "trust_untrusted", "trust_blocked", "observed_in_future", "expired",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def semantic_payload(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "suite_id": report.get("suite_id"),
        "suite_version": report.get("suite_version"),
        "suite_sha256": report.get("suite_sha256"),
        "index_sha256": report.get("index_sha256"),
        "index_manifest": report.get("index_manifest"),
        "inventory": report.get("inventory"),
        "metrics": {
            key: value
            for key, value in report.get("metrics", {}).items()
            if key != "mean_retrieval_latency_ms"
        },
        "records": [
            {key: value for key, value in record.items() if key != "retrieval_latency_ms"}
            for record in report.get("records", [])
        ],
    }


def legacy_semantic_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Return the exact semantic payload used by released report schema 1.0."""

    return {
        "suite_sha256": report.get("suite_sha256"),
        "index_sha256": report.get("index_sha256"),
        "inventory": report.get("inventory"),
        "metrics": {
            key: value
            for key, value in report.get("metrics", {}).items()
            if "latency" not in key
        },
        "records": [
            {key: value for key, value in record.items() if key != "retrieval_latency_ms"}
            for record in report.get("records", [])
        ],
    }


def _number(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return False
    if not math.isfinite(numeric):
        return False
    if minimum is not None and numeric < minimum:
        return False
    if maximum is not None and numeric > maximum:
        return False
    return True


def _same_number(left: Any, right: Any) -> bool:
    return _number(left) and _number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def _digest(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _string_list(value: Any, *, identifiers: bool = False, unique: bool = False) -> bool:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return False
    if identifiers and any(not _identifier(item) for item in value):
        return False
    return not unique or len(value) == len(set(value))


def _verify_legacy_report(report: dict[str, Any]) -> bool:
    if set(report) != LEGACY_REPORT_FIELDS or report.get("schema_version") != "1.0":
        return False
    if not _identifier(report.get("suite_id")):
        return False
    version = report.get("suite_version")
    if not isinstance(version, str) or SEMVER_RE.fullmatch(version) is None:
        return False
    if not _digest(report.get("suite_sha256")) or not _digest(report.get("index_sha256")):
        return False
    if not isinstance(report.get("inventory"), dict) or not isinstance(report.get("metrics"), dict):
        return False
    records = report.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_QUESTIONS:
        return False
    question_ids: list[str] = []
    for record in records:
        if not isinstance(record, dict) or not _identifier(record.get("question_id")):
            return False
        if not _string_list(record.get("failures")):
            return False
        question_ids.append(record["question_id"])
    if len(question_ids) != len(set(question_ids)):
        return False
    expected = report.get("semantic_sha256")
    try:
        return _digest(expected) and expected == canonical_sha256(legacy_semantic_payload(report))
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False


def _verify_inventory(inventory: Any, suite_sha256: str) -> tuple[bool, dict[str, dict[str, Any]]]:
    if not isinstance(inventory, dict) or set(inventory) != INVENTORY_FIELDS:
        return False, {}
    if inventory.get("suite_sha256") != suite_sha256:
        return False, {}
    documents = inventory.get("documents")
    count = inventory.get("document_count")
    valid_count = inventory.get("valid_document_count")
    if (
        not isinstance(documents, list)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= MAX_DOCUMENTS
        or count != len(documents)
        or isinstance(valid_count, bool)
        or not isinstance(valid_count, int)
        or not 0 <= valid_count <= count
    ):
        return False, {}
    rows: dict[str, dict[str, Any]] = {}
    hashes: dict[str, list[str]] = {}
    for row in documents:
        if not isinstance(row, dict) or set(row) != INVENTORY_DOCUMENT_FIELDS:
            return False, {}
        source_id = row.get("source_id")
        errors = row.get("errors")
        if not _identifier(source_id) or source_id in rows or not _digest(row.get("sha256")):
            return False, {}
        if not _string_list(errors, unique=True) or any(error not in DOCUMENT_ERROR_CODES for error in errors):
            return False, {}
        if not isinstance(row.get("valid"), bool) or row["valid"] != (not errors):
            return False, {}
        rows[source_id] = row
        hashes.setdefault(row["sha256"], []).append(source_id)
    if valid_count != sum(1 for row in documents if row["valid"]):
        return False, {}
    if not _same_number(inventory.get("fresh_coverage"), valid_count / count):
        return False, {}
    duplicates = inventory.get("duplicates")
    expected_duplicates = sorted(sorted(ids) for ids in hashes.values() if len(ids) > 1)
    if duplicates != expected_duplicates:
        return False, {}
    return True, rows


def _verify_claim(claim: Any) -> bool:
    if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
        return False
    if not isinstance(claim.get("text"), str) or not claim["text"].strip():
        return False
    if not _string_list(claim.get("citations"), identifiers=True, unique=True) or not claim["citations"]:
        return False
    if not isinstance(claim.get("supported"), bool) or not _number(
        claim.get("token_coverage"), minimum=0.0, maximum=1.0
    ):
        return False
    if not _string_list(claim.get("reasons"), unique=True):
        return False
    if claim["supported"]:
        return claim["token_coverage"] >= 0.75 and not claim["reasons"]
    return bool(claim["reasons"])


def _verify_metrics_shape(metrics: Any) -> bool:
    if not isinstance(metrics, dict) or set(metrics) != METRIC_FIELDS:
        return False
    ratios = METRIC_FIELDS - {
        "pass_rate_ci95", "mean_retrieval_latency_ms", "retrieval_k", "retrieval_strategy",
    }
    if any(not _number(metrics.get(key), minimum=0.0, maximum=1.0) for key in ratios):
        return False
    if not _number(metrics.get("mean_retrieval_latency_ms"), minimum=0.0):
        return False
    retrieval_k = metrics.get("retrieval_k")
    if isinstance(retrieval_k, bool) or not isinstance(retrieval_k, int) or not 1 <= retrieval_k <= 20:
        return False
    if metrics.get("retrieval_strategy") not in RETRIEVAL_STRATEGIES:
        return False
    interval = metrics.get("pass_rate_ci95")
    if not isinstance(interval, dict) or set(interval) != {"mean", "lower", "upper", "iterations", "seed"}:
        return False
    if any(not _number(interval.get(key), minimum=0.0, maximum=1.0) for key in ("mean", "lower", "upper")):
        return False
    return interval.get("iterations") == 1000 and interval.get("seed") == 17


def _verify_new_report(report: dict[str, Any]) -> bool:
    if set(report) != REPORT_FIELDS or report.get("schema_version") != "2.0":
        return False
    if not _identifier(report.get("suite_id")):
        return False
    version = report.get("suite_version")
    if not isinstance(version, str) or SEMVER_RE.fullmatch(version) is None:
        return False
    suite_sha256 = report.get("suite_sha256")
    index_sha256 = report.get("index_sha256")
    manifest = report.get("index_manifest")
    if not _digest(suite_sha256) or not _digest(index_sha256) or not verify_manifest(manifest):
        return False
    if manifest["suite_sha256"] != suite_sha256 or manifest["manifest_sha256"] != index_sha256:
        return False
    inventory_valid, document_rows = _verify_inventory(report.get("inventory"), suite_sha256)
    if not inventory_valid:
        return False
    valid_sources = {source_id for source_id, row in document_rows.items() if row["valid"]}
    manifest_chunks = {chunk["chunk_id"]: chunk for chunk in manifest["chunks"]}
    if {chunk["source_id"] for chunk in manifest["chunks"]} != valid_sources:
        return False
    for chunk in manifest["chunks"]:
        source = document_rows.get(chunk["source_id"])
        if source is None or chunk["document_sha256"] != source["sha256"]:
            return False

    metrics = report.get("metrics")
    if not _verify_metrics_shape(metrics):
        return False
    if metrics["retrieval_strategy"] != manifest["strategy"]:
        return False
    records = report.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_QUESTIONS:
        return False
    question_ids: set[str] = set()
    claims_flat: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
            return False
        question_id = record.get("question_id")
        if not _identifier(question_id) or question_id in question_ids:
            return False
        question_ids.add(question_id)
        if not isinstance(record.get("adversarial"), bool):
            return False
        expected_list = record.get("expected_source_ids")
        if not _string_list(expected_list, identifiers=True, unique=True):
            return False
        expected_sources = set(expected_list)
        if not expected_sources.issubset(document_rows):
            return False
        if record.get("retrieval_strategy") != metrics["retrieval_strategy"]:
            return False
        if not _number(record.get("retrieval_latency_ms"), minimum=0.0):
            return False
        retrieved = record.get("retrieved")
        if not isinstance(retrieved, list) or len(retrieved) > metrics["retrieval_k"]:
            return False
        retrieved_ids: set[str] = set()
        ranked_sources: list[str] = []
        previous_score: float | None = None
        for row in retrieved:
            if not isinstance(row, dict) or set(row) != RETRIEVED_FIELDS:
                return False
            chunk_id = row.get("chunk_id")
            source_id = row.get("source_id")
            manifest_chunk = manifest_chunks.get(chunk_id)
            if (
                not isinstance(chunk_id, str)
                or chunk_id in retrieved_ids
                or not _identifier(source_id)
                or manifest_chunk is None
                or manifest_chunk["source_id"] != source_id
                or isinstance(row.get("ordinal"), bool)
                or not isinstance(row.get("ordinal"), int)
                or manifest_chunk["ordinal"] != row.get("ordinal")
                or manifest_chunk["document_sha256"] != row.get("document_sha256")
                or not isinstance(row.get("text"), str)
                or hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != manifest_chunk["text_sha256"]
            ):
                return False
            if any(not _number(row.get(key), minimum=0.0) for key in ("score", "lexical_score", "vector_score")):
                return False
            if row["score"] <= 0:
                return False
            current_score = float(row["score"])
            if previous_score is not None and current_score > previous_score:
                return False
            previous_score = current_score
            retrieved_ids.add(chunk_id)
            ranked_sources.append(source_id)
        retrieved_sources = set(ranked_sources)
        expected_recall = 1.0 if not expected_sources else len(expected_sources & retrieved_sources) / len(expected_sources)
        expected_precision = round(
            precision_at_k(expected_sources, ranked_sources, metrics["retrieval_k"]), 6
        )
        expected_rr = round(reciprocal_rank(expected_sources, ranked_sources), 6)
        expected_ndcg = round(ndcg_at_k(expected_sources, ranked_sources, metrics["retrieval_k"]), 6)
        if not all(
            (
                _same_number(record.get("recall_at_k"), expected_recall),
                _same_number(record.get("precision_at_k"), expected_precision),
                _same_number(record.get("reciprocal_rank"), expected_rr),
                _same_number(record.get("ndcg_at_k"), expected_ndcg),
            )
        ):
            return False

        claims = record.get("claims")
        if not isinstance(claims, list) or any(not _verify_claim(claim) for claim in claims):
            return False
        claims_flat.extend(claims)
        citations = {citation for claim in claims for citation in claim["citations"]}
        expected_citation_errors = sorted(citations - retrieved_sources)
        expected_citations_valid = not expected_citation_errors and citations.issubset(valid_sources)
        expected_citation_precision, expected_citation_recall = citation_scores(
            citations, expected_sources, valid_sources, retrieved_sources
        )
        if record.get("citation_errors") != expected_citation_errors:
            return False
        if record.get("citations_valid") is not expected_citations_valid:
            return False
        if not _same_number(record.get("citation_precision"), round(expected_citation_precision, 6)):
            return False
        if not _same_number(record.get("citation_recall"), round(expected_citation_recall, 6)):
            return False
        no_answer = not expected_sources
        expected_no_answer_correct = (not claims) if no_answer else bool(claims)
        expected_grounded = all(claim["supported"] for claim in claims) if claims else no_answer
        expected_passed = (
            expected_recall == 1.0
            and expected_citations_valid
            and expected_no_answer_correct
            and expected_grounded
        )
        failures: list[str] = []
        if expected_recall < 1.0:
            failures.append("retrieval_miss")
        if not expected_citations_valid:
            failures.append("invalid_citation")
        if not expected_no_answer_correct:
            failures.append("no_answer_mismatch")
        if not expected_grounded:
            failures.append("ungrounded_claim")
        if (
            record.get("no_answer_correct") is not expected_no_answer_correct
            or record.get("grounded") is not expected_grounded
            or record.get("passed") is not expected_passed
            or record.get("failures") != failures
            or any(failure not in FAILURE_CODES for failure in failures)
        ):
            return False

    answerable = [record for record in records if record["expected_source_ids"]]
    no_answer_records = [record for record in records if not record["expected_source_ids"]]
    expected_metrics = {
        "recall_at_k": sum(record["recall_at_k"] for record in answerable) / len(answerable) if answerable else 1.0,
        "precision_at_k": sum(record["precision_at_k"] for record in answerable) / len(answerable) if answerable else 1.0,
        "mean_reciprocal_rank": sum(record["reciprocal_rank"] for record in answerable) / len(answerable) if answerable else 1.0,
        "ndcg_at_k": sum(record["ndcg_at_k"] for record in answerable) / len(answerable) if answerable else 1.0,
        "groundedness": sum(1 for claim in claims_flat if claim["supported"]) / len(claims_flat) if claims_flat else 1.0,
        "citation_precision": sum(record["citation_precision"] for record in answerable) / len(answerable) if answerable else 1.0,
        "citation_recall": sum(record["citation_recall"] for record in answerable) / len(answerable) if answerable else 1.0,
        "no_answer_accuracy": sum(1 for record in no_answer_records if record["no_answer_correct"]) / len(no_answer_records) if no_answer_records else 1.0,
        "fresh_coverage": report["inventory"]["fresh_coverage"],
        "pass_rate": sum(1 for record in records if record["passed"]) / len(records),
        "adversarial_pass_rate": (
            sum(1 for record in records if record["adversarial"] and record["passed"])
            / sum(1 for record in records if record["adversarial"])
            if any(record["adversarial"] for record in records)
            else 1.0
        ),
        "mean_retrieval_latency_ms": sum(record["retrieval_latency_ms"] for record in records) / len(records),
    }
    if any(not _same_number(metrics.get(key), value) for key, value in expected_metrics.items()):
        return False
    if metrics["pass_rate_ci95"] != bootstrap_mean_ci(
        [1.0 if record["passed"] else 0.0 for record in records], seed=17
    ):
        return False
    expected = report.get("semantic_sha256")
    try:
        return _digest(expected) and expected == canonical_sha256(semantic_payload(report))
    except (AttributeError, OverflowError, TypeError, ValueError):
        return False


def verify_report(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict):
        return False
    if report.get("schema_version") == "1.0":
        return _verify_legacy_report(report)
    if report.get("schema_version") == "2.0":
        return _verify_new_report(report)
    return False


def load_report(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise ContractError(f"report does not exist: {target}")
    try:
        with target.open("rb") as stream:
            encoded = stream.read(MAX_REPORT_BYTES + 1)
        if len(encoded) > MAX_REPORT_BYTES:
            raise ContractError(f"report exceeds {MAX_REPORT_BYTES} bytes")
        raw = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except ContractError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(f"cannot read report: {target}") from exc
    if not isinstance(raw, dict) or not verify_report(raw):
        raise ContractError("report semantic SHA verification failed")
    return raw


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    if target.exists() and target.is_dir():
        raise ContractError("report path cannot be a directory")
    if not verify_report(report):
        raise ContractError("refusing to write an invalid report")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if len(payload.encode("utf-8")) > MAX_REPORT_BYTES:
        raise ContractError(f"report exceeds {MAX_REPORT_BYTES} bytes")
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
    load_report(target)


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (OverflowError, TypeError, ValueError):
        return "n/a"


def _markdown_cell(value: Any) -> str:
    return (
        escape(str(value), quote=False)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def render_markdown(report: dict[str, Any]) -> str:
    if not verify_report(report):
        raise ContractError("cannot render an unverified report")
    metrics = report["metrics"]
    lines = [
        "# RAG Quality Bench report",
        "",
        f"- Suite: `{_markdown_cell(report.get('suite_id'))}`",
        f"- Suite version: `{_markdown_cell(report.get('suite_version'))}`",
        f"- Retrieval: `{_markdown_cell(metrics.get('retrieval_strategy'))}`",
        f"- Semantic SHA-256: `{_markdown_cell(report.get('semantic_sha256'))}`",
        "",
        "## Quality metrics",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Pass rate | {_percent(metrics.get('pass_rate'))} |",
        f"| Recall@k | {_percent(metrics.get('recall_at_k'))} |",
        f"| Precision@k | {_percent(metrics.get('precision_at_k'))} |",
        f"| MRR | {_percent(metrics.get('mean_reciprocal_rank'))} |",
        f"| nDCG@k | {_percent(metrics.get('ndcg_at_k'))} |",
        f"| Groundedness | {_percent(metrics.get('groundedness'))} |",
        f"| Citation precision | {_percent(metrics.get('citation_precision'))} |",
        f"| Citation recall | {_percent(metrics.get('citation_recall'))} |",
        f"| No-answer accuracy | {_percent(metrics.get('no_answer_accuracy'))} |",
        f"| Fresh coverage | {_percent(metrics.get('fresh_coverage'))} |",
        "",
        "## Question outcomes",
        "",
        "| Question | Passed | Recall | MRR | nDCG | Failures |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in report.get("records", []):
        lines.append(
            "| {question} | {passed} | {recall} | {mrr} | {ndcg} | {failures} |".format(
                question=_markdown_cell(record.get("question_id")),
                passed="yes" if record.get("passed") else "no",
                recall=_percent(record.get("recall_at_k")),
                mrr=_percent(record.get("reciprocal_rank")),
                ndcg=_percent(record.get("ndcg_at_k")),
                failures=_markdown_cell(", ".join(record.get("failures", [])) or "none"),
            )
        )
    lines.extend(
        [
            "",
            "This report preserves failed and adversarial cases. It is generated from synthetic or caller-supplied data and does not prove model quality outside this suite.",
            "",
        ]
    )
    return "\n".join(lines)


def render_html(report: dict[str, Any]) -> str:
    if not verify_report(report):
        raise ContractError("cannot render an unverified report")
    metrics = report["metrics"]
    metric_rows = [
        ("Pass rate", metrics.get("pass_rate")),
        ("Recall@k", metrics.get("recall_at_k")),
        ("Precision@k", metrics.get("precision_at_k")),
        ("MRR", metrics.get("mean_reciprocal_rank")),
        ("nDCG@k", metrics.get("ndcg_at_k")),
        ("Groundedness", metrics.get("groundedness")),
        ("Citation precision", metrics.get("citation_precision")),
        ("Citation recall", metrics.get("citation_recall")),
        ("No-answer accuracy", metrics.get("no_answer_accuracy")),
        ("Fresh coverage", metrics.get("fresh_coverage")),
    ]
    cards = "".join(
        f'<section class="metric"><span>{escape(label)}</span><strong>{escape(_percent(value))}</strong></section>'
        for label, value in metric_rows
    )
    rows = "".join(
        "<tr><td>{}</td><td class=\"{}\">{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(record.get("question_id", ""))),
            "pass" if record.get("passed") else "fail",
            "PASS" if record.get("passed") else "FAIL",
            escape(_percent(record.get("recall_at_k"))),
            escape(_percent(record.get("ndcg_at_k"))),
            escape(", ".join(record.get("failures", [])) or "none"),
        )
        for record in report.get("records", [])
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG Quality Bench - {escape(str(report.get('suite_id', 'report')))}</title>
<style>
:root{{--bg:#0b1020;--panel:#151d32;--line:#293653;--text:#ecf1ff;--muted:#9eabc7;--good:#53d39b;--bad:#ff718b;--accent:#78a6ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1120px;margin:auto;padding:48px 24px}} h1{{font-size:clamp(2rem,5vw,4rem);margin:.2rem 0}} .meta{{color:var(--muted)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:28px 0}}
.metric{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}} .metric span{{display:block;color:var(--muted)}} .metric strong{{font-size:1.7rem}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:14px;overflow:hidden}} th,td{{padding:12px;border-bottom:1px solid var(--line);text-align:left}} .pass{{color:var(--good)}} .fail{{color:var(--bad)}} code{{color:var(--accent);word-break:break-all}}
</style></head><body><main>
<p class="meta">OFFLINE · DETERMINISTIC · EVIDENCE FIRST</p><h1>RAG Quality Bench</h1>
<p>Suite <strong>{escape(str(report.get('suite_id')))}</strong> · retrieval <strong>{escape(str(metrics.get('retrieval_strategy')))}</strong></p>
<p class="meta">Semantic SHA-256 <code>{escape(str(report.get('semantic_sha256')))}</code></p>
<div class="grid">{cards}</div>
<h2>Question outcomes</h2><table><thead><tr><th>Question</th><th>Verdict</th><th>Recall</th><th>nDCG</th><th>Failures</th></tr></thead><tbody>{rows}</tbody></table>
<p class="meta">Failed and adversarial cases are preserved. Results apply only to this versioned suite.</p>
</main></body></html>"""


def export_report(path: str | Path, report: dict[str, Any], *, format: str) -> None:
    if format not in {"markdown", "html"}:
        raise ContractError("export format must be markdown or html")
    target = Path(path)
    if target.exists() and target.is_dir():
        raise ContractError("export path cannot be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = render_markdown(report) if format == "markdown" else render_html(report)
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
