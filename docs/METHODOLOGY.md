# Methodology

## Fair comparisons

Compare suites with the same questions, retrieval `k`, evaluation date, and source contracts. Record every configuration change. A different strategy, chunk size, overlap, hybrid weight, or corpus produces a distinct suite/index digest and must be reviewed as a different candidate.

## Judges

- Retrieval recall, precision, reciprocal rank, and nDCG use expected source IDs. Repeated chunks from one expected source count once for relevance so chunk density cannot inflate source-level ranking quality.
- Citation validity requires cited sources to be valid and present in retrieval results.
- Citation precision divides expected, valid, retrieved citations by all citations. Citation recall divides the same eligible citations by all expected sources; an expected source that was not retrieved receives no citation credit.
- Groundedness requires at least 75 percent claim-token coverage in cited source content.
- No-answer accuracy compares explicit abstention with the question contract.
- Adversarial pass rate is isolated from the global pass rate so counter-examples remain visible.
- The pass-rate 95% interval uses a fixed-seed bootstrap. It is a reproducibility aid, not a substitute for statistical power analysis.

These transparent local rules are intentionally limited. Token overlap, BM25, TF-IDF, and feature hashing do not prove semantic entailment. Results must report that limitation and preserve adversarial failures.

## Sweeps

Each sweep dimension is capped at 32 inputs and enumeration stops before a 33rd valid configuration. The quality score weights pass rate (40%), recall (20%), nDCG (15%), groundedness (15%), and no-answer accuracy (10%). Ties fall back to pass rate, recall, then stable run ID. A winning configuration is a candidate for review, not automatic evidence that it generalizes.

## Reproducibility

Latency depends on the machine, so it is reported but excluded from semantic hashes. All logical records, source hashes, retrieval results, index metadata, and non-latency metrics are verified. Run the same suite on representative data and keep the raw JSON alongside any rendered report.
