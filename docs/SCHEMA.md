# Suite Schema 1.0

A suite declares a semantic version, fixed evaluation date, chunk configuration, retrieval `k`, optional retrieval strategy (`overlap`, `bm25`, `tfidf`, or `hybrid`), optional hybrid weight from 0 to 1, source contracts, questions, and candidate responses. Omitted retrieval fields preserve the `overlap` and `0.5` defaults.

Each document requires source ID, URL, license, observed date, trust state, content, and SHA-256. Optional expiration makes freshness deterministic.

Each answerable question requires expected source IDs and a response with one or more claims. Each claim requires explicit source citations. A no-answer question expects no source and an abstaining response.

Limits: 200 documents, 500 questions, 200,000 characters per document, chunk size 8-500, and retrieval `k` 1-20.

Unknown fields, invalid booleans, non-finite numeric values, duplicate source/question/expected-source/citation IDs, and references to unknown expected sources fail validation. To preserve released schema-1.0 identities, the canonical suite SHA omits the retrieval pair only when both values are the legacy defaults (`overlap`, `0.5`); explicit and omitted defaults therefore have the same digest. Any non-default retrieval configuration includes both fields in the digest.

## Report Schema 2.0

New reports use `schema_version: "2.0"`. The semantic SHA covers report and suite identity, the complete self-hashed index manifest, inventory, non-latency metrics, and every non-latency question field. Verification is structural and logical: nested fields are exact, identifiers and digests are bounded, retrieved text must match manifest hashes, per-question metrics are recomputed, aggregate metrics must match retained records, and the suite/index/manifest digests must agree.

Machine-dependent retrieval latency remains outside the semantic SHA but must be finite, non-negative, and internally aggregated correctly.

Released report schema `1.0` remains loadable and exportable with its original field set and semantic-hash algorithm. Schema 1.0 lacks an embedded manifest, so it cannot provide the additional cross-link proofs of schema 2.0. Loaders verify legacy evidence in place and never relabel it as 2.0.
