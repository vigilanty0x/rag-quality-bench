# Security Policy

## Supported versions

Security fixes target the latest release.

## Reporting

Report vulnerabilities privately through GitHub's security reporting flow. Do not open a public issue containing exploit details or sensitive data.

## Security model

- Suite, report, and manifest JSON are size-bounded; ambiguous duplicate keys are rejected where documents are loaded.
- Source content is accepted only when its SHA-256, trust, and freshness checks pass.
- Unknown fields fail validation to expose schema drift.
- Reports are deeply validated before replacement, written atomically, and verified again after reopening, so a rejected report cannot overwrite prior valid evidence.
- Report schema 2.0 semantic hashes bind report/suite identity and logical results. Verification checks nested field sets, recomputed metrics, embedded manifest validity, retrieved text hashes, and suite/index/manifest digest links. Schema 1.0 remains supported with its documented legacy boundary.
- Index manifests omit source text, are size-bounded, are written atomically, and verify their canonical self-hash.
- Chunk counts and retrieval configuration dimensions are rejected before unbounded materialization or Cartesian expansion.
- CI actions are pinned to immutable commits, and the release build frontend/backend versions are exact and attested before wheel/sdist creation.
- The runtime performs no network requests and executes no corpus content.

This tool is an evaluation aid, not a sandbox for hostile files. Keep untrusted datasets outside privileged environments. Self-hashes provide integrity, not provenance or identity; authenticate artifacts through a trusted release or an external signing system when that distinction matters.
