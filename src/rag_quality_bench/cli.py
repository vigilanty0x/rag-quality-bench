"""Bounded command-line interface."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.resources import files
import json
from pathlib import Path
import sys
from typing import Any, BinaryIO

from .engine import BenchmarkEngine, compare_reports
from .experiments import ExperimentError, run_sweep, write_sweep
from .models import BenchmarkSuite, ContractError, MAX_SUITE_BYTES
from .probes import functional_probe, liveness_probe, readiness_probe
from .reporting import export_report, load_report, write_report
from .retrieval import RetrievalError, load_manifest, write_manifest


def _suite_from_stream(stream: BinaryIO, *, label: str) -> BenchmarkSuite:
    raw = stream.read(MAX_SUITE_BYTES + 1)
    if len(raw) > MAX_SUITE_BYTES:
        raise ContractError("suite JSON exceeds 5 MB")
    try:
        return BenchmarkSuite.from_json(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ContractError(f"suite must be UTF-8 JSON: {label}") from exc


def _load_suite(path: str) -> BenchmarkSuite:
    target = Path(path)
    if not target.is_file():
        raise ContractError(f"suite does not exist: {target}")
    with target.open("rb") as stream:
        return _suite_from_stream(stream, label=str(target))


def _load_demo_suite() -> BenchmarkSuite:
    resource = files("rag_quality_bench").joinpath("data", "suite.json")
    with resource.open("rb") as stream:
        return _suite_from_stream(stream, label="bundled demo suite")


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-quality-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a versioned benchmark suite")
    validate.add_argument("--suite", required=True)
    run = commands.add_parser("run", help="run a benchmark and preserve all failures")
    run.add_argument("--suite", required=True)
    run.add_argument("--output")
    run.add_argument("--minimum-pass-rate", type=float)
    run.add_argument("--strategy", choices=["overlap", "bm25", "tfidf", "hybrid"])
    verify = commands.add_parser("verify", help="verify a written report")
    verify.add_argument("--report", required=True)
    compare = commands.add_parser("compare", help="diff two verified reports")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    inventory = commands.add_parser("inventory", help="show trust, freshness, and duplicates")
    inventory.add_argument("--suite", required=True)
    probe = commands.add_parser("probe", help="run a health or counter-proof probe")
    probe.add_argument("--level", required=True, choices=["liveness", "readiness", "functional"])
    demo = commands.add_parser("demo", help="run the public synthetic example")
    demo.add_argument("--output", default="reports/demo.json")

    index = commands.add_parser("index", help="write a redacted, content-bound index manifest")
    index.add_argument("--suite", required=True)
    index.add_argument("--strategy", choices=["overlap", "bm25", "tfidf", "hybrid"])
    index.add_argument("--output", required=True)
    verify_index = commands.add_parser("verify-index", help="verify an index manifest self-hash")
    verify_index.add_argument("--manifest", required=True)

    sweep = commands.add_parser("sweep", help="compare a bounded grid of retrieval configurations")
    sweep.add_argument("--suite", required=True)
    sweep.add_argument("--strategies", default="overlap,bm25,tfidf,hybrid")
    sweep.add_argument("--chunk-sizes", default="12,20,40")
    sweep.add_argument("--overlaps", default="0,2")
    sweep.add_argument("--output")

    export = commands.add_parser("export", help="render a verified report as Markdown or standalone HTML")
    export.add_argument("--report", required=True)
    export.add_argument("--format", required=True, choices=["markdown", "html"])
    export.add_argument("--output", required=True)
    return parser


def _csv_strings(value: str, label: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or len(items) != len(set(items)):
        raise ContractError(f"{label} must be a non-empty comma-separated list without duplicates")
    return items


def _csv_ints(value: str, label: str) -> list[int]:
    raw = _csv_strings(value, label)
    try:
        return [int(item) for item in raw]
    except ValueError as exc:
        raise ContractError(f"{label} must contain integers") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            suite = _load_suite(args.suite)
            _print({
                "valid": True,
                "suite_id": suite.suite_id,
                "version": suite.version,
                "documents": len(suite.documents),
                "questions": len(suite.questions),
                "suite_sha256": suite.digest,
            })
            return 0
        if args.command == "run":
            if args.minimum_pass_rate is not None and not 0 <= args.minimum_pass_rate <= 1:
                raise ContractError("minimum pass rate must be between 0 and 1")
            suite = _load_suite(args.suite)
            if args.strategy:
                suite = replace(suite, retrieval_strategy=args.strategy)
            report = BenchmarkEngine(suite).run()
            if args.output:
                write_report(args.output, report)
            _print(report)
            return int(args.minimum_pass_rate is not None and report["metrics"]["pass_rate"] < args.minimum_pass_rate)
        if args.command == "verify":
            report = load_report(args.report)
            _print({
                "valid": True,
                "schema_version": report["schema_version"],
                "semantic_sha256": report["semantic_sha256"],
            })
            return 0
        if args.command == "compare":
            _print(compare_reports(load_report(args.baseline), load_report(args.candidate)))
            return 0
        if args.command == "inventory":
            _print(BenchmarkEngine(_load_suite(args.suite)).inventory())
            return 0
        if args.command == "probe":
            probes = {"liveness": liveness_probe, "readiness": readiness_probe, "functional": functional_probe}
            result = probes[args.level]()
            _print(result)
            return 0 if result["ok"] else 1
        if args.command == "demo":
            report = BenchmarkEngine(_load_demo_suite()).run()
            write_report(args.output, report)
            _print({
                "output": args.output,
                "metrics": report["metrics"],
                "failures_preserved": sum(len(row["failures"]) for row in report["records"]),
                "semantic_sha256": report["semantic_sha256"],
            })
            return 0
        if args.command == "index":
            suite = _load_suite(args.suite)
            if args.strategy:
                suite = replace(suite, retrieval_strategy=args.strategy)
            manifest = BenchmarkEngine(suite).index_manifest()
            write_manifest(args.output, manifest)
            _print({"valid": True, "output": args.output, **manifest})
            return 0
        if args.command == "verify-index":
            manifest = load_manifest(args.manifest)
            _print({"valid": True, "manifest_sha256": manifest["manifest_sha256"], "chunks": manifest["chunk_count"]})
            return 0
        if args.command == "sweep":
            result = run_sweep(
                _load_suite(args.suite),
                strategies=_csv_strings(args.strategies, "strategies"),
                chunk_sizes=_csv_ints(args.chunk_sizes, "chunk sizes"),
                overlaps=_csv_ints(args.overlaps, "overlaps"),
            )
            if args.output:
                write_sweep(args.output, result)
            _print(result)
            return 0
        if args.command == "export":
            report = load_report(args.report)
            export_report(args.output, report, format=args.format)
            _print({"written": args.output, "format": args.format, "semantic_sha256": report["semantic_sha256"]})
            return 0
    except (ContractError, ExperimentError, RetrievalError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2
