# Architecture

The dependency direction is deliberately small:

1. `models` validates versioned, bounded public contracts and canonicalizes their digest.
2. `retrieval` builds a bounded in-memory overlap, BM25, TF-IDF, or hybrid index and emits a redacted self-hashed manifest.
3. `metrics` implements transparent ranking, citation, and seeded bootstrap calculations.
4. `engine` rejects invalid sources, chunks valid content, retrieves, evaluates every question, and retains failures.
5. `experiments` runs a bounded configuration grid and selects a winner with a documented deterministic score.
6. `reporting` atomically writes/verifies JSON evidence and renders escaped Markdown or standalone HTML.
7. `probes` proves process health separately from real functional behavior.
8. `cli` exposes the workflow without provider credentials, subprocesses, or network clients.

Every retrieved chunk carries its source ID, document SHA, ordinal, and text. Every claim lists explicit source citations. Aggregate metrics are derived from retained question records.

## Evidence boundaries

The suite SHA binds source contracts and source text. The index manifest SHA separately binds strategy, chunk settings, document hashes, chunk IDs, and per-chunk text hashes while omitting raw text. Report schema 2.0 binds report identity, suite/index identity, inventory, logical metrics, and question records while deliberately excluding machine-dependent latency. Its verifier recomputes aggregate metrics, validates every nested record and manifest, and enforces the suite/index/manifest digest links. Released schema 1.0 reports remain readable through their original semantic payload. Unknown fields and ambiguous duplicate JSON keys are rejected. None of these hashes is a signature: trust still depends on how the file was obtained.

## Resource bounds

Suite, document, question, retrieval, manifest, and sweep sizes are capped. The engine counts chunks and rejects a corpus exceeding 100,000 before materializing any chunk; the generic index constructor stops consuming an iterable at the same boundary. Sweep dimensions are each capped at 32 and configuration enumeration stops before a 33rd valid run. Corpus text is treated as data and is never executed or downloaded.
