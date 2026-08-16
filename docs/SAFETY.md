# Safety and Public Data Boundary

RAG Quality Bench never downloads sources or executes content. Corpus authors supply already-approved public or synthetic text with provenance and license metadata.

The evaluator rejects a document if its content hash differs, its trust is not `trusted`, it is expired, or its observation date is in the future. Rejection is visible in inventory and may lower retrieval recall; it never silently becomes success.

Use reserved example domains and synthetic fixtures in public contributions. Keep private documents, customer names, internal URLs, secrets, and production index metadata out of the repository.

Redacted index manifests omit source text but retain source IDs, document hashes, chunk IDs, and token counts. Those fields may still reveal sensitive naming or corpus shape, so inspect manifests before sharing them. Hash verification detects accidental or malicious mutation after creation; it does not authenticate an author.
