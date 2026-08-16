# RAG Quality Bench

RAG Quality Bench is a dependency-free, offline evaluation laboratory for retrieval-augmented generation systems. It compares four reproducible retrieval strategies, scores answer and ranking quality, records a content-bound index manifest, sweeps bounded configurations, and produces verifiable JSON plus human-readable Markdown or HTML evidence.

It keeps the complete trace from source contract to chunk, retrieval rank, claim, citation, and verdict. Invalid, expired, blocked, future-dated, or hash-mismatched sources are rejected fail-closed. Individual failures remain visible instead of disappearing behind aggregate scores.

## Quick start

```bash
python -m pip install .
rag-quality-bench validate --suite examples/suite.json
rag-quality-bench run --suite examples/suite.json --output reports/demo.json
rag-quality-bench verify --report reports/demo.json
rag-quality-bench index --suite examples/suite.json --strategy hybrid --output reports/index.json
rag-quality-bench sweep --suite examples/suite.json --strategies overlap,bm25,tfidf,hybrid --chunk-sizes 12,20 --overlaps 0,2 --output reports/sweep.json
rag-quality-bench export --report reports/demo.json --format html --output reports/demo.html
rag-quality-bench probe --level functional
```

The example is synthetic and runs without a model account, network request, private corpus, or vector database.

## Evaluation surface

- recall@k, precision@k, reciprocal rank, and nDCG@k against expected source IDs
- claim groundedness using explicit cited evidence
- correct abstention on no-answer questions
- citation precision and recall against retrieved, trusted, fresh sources
- freshness coverage, adversarial pass rate, duplicates, pass rate, and measured retrieval latency
- seeded bootstrap confidence intervals for the benchmark pass rate
- semantic differences between two verified reports

Measured latency is reported but excluded from the semantic evidence SHA so the same logical result can be verified across clean machines.

New reports use report schema `2.0`, whose verifier checks nested records, aggregate metrics, the embedded index manifest, and all suite/index digest links. The loader remains backward-compatible with released schema `1.0` reports; legacy reports retain their original hash algorithm and are never silently rewritten.

## Retrieval strategies

- `overlap`: transparent query-token coverage baseline.
- `bm25`: bounded Okapi BM25 lexical ranking.
- `tfidf`: deterministic TF-IDF cosine ranking.
- `hybrid`: BM25 combined with a fixed-size feature-hashed vector score.

All strategies run locally with no model, service, embedding endpoint, or hidden state. The hybrid vector is deliberately a reproducible experimental baseline, not a semantic embedding model.

## Commands

- `validate`: validate the bounded versioned suite contract.
- `inventory`: show freshness, trust, hash status, and duplicate content.
- `run`: evaluate every question and optionally write an atomic verified report.
- `verify`: validate report structure and logical aggregates, enforce digest links, and recompute the semantic SHA.
- `compare`: show metric deltas and changed question outcomes.
- `index` / `verify-index`: write and verify a redacted index manifest that binds configuration and chunk hashes without publishing source text.
- `sweep`: compare at most 32 strategy/chunk configurations and select the best run by a documented deterministic score.
- `export`: turn a verified JSON report into safe Markdown or standalone HTML.
- `probe`: run separate liveness, readiness, or functional counter-proof checks.
- `demo`: run the bundled synthetic corpus and preserve the intentional failure.

Use `--minimum-pass-rate` with `run` to make CI fail when the report falls below an explicit gate.

## Public boundary

Only generic implementation code and synthetic `example.invalid` fixtures belong here. Do not add client names, private documents, credentials, internal topology, production URLs, or proprietary evaluation data. Index manifests intentionally contain source identifiers and hashes, so review those identifiers before publication. See [SECURITY.md](SECURITY.md), [docs/METHODOLOGY.md](docs/METHODOLOGY.md), and [AI_ASSISTANCE.md](AI_ASSISTANCE.md).

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/check.py
python -m pip install --upgrade -r requirements-build.txt
python -m pip wheel . --no-deps --no-build-isolation --wheel-dir dist
python -c 'from setuptools.build_meta import build_sdist; print(build_sdist("dist"))'
```

`requirements-build.txt` is the authoritative release toolchain. CI attests those exact versions before creating artifacts, installs the built wheel rather than the checkout, and executes the included tests from the sdist.

Licensed under Apache-2.0.
