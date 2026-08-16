"""Deterministic, offline retrieval strategies and redacted index manifests."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

from .models import RETRIEVAL_STRATEGIES, canonical_sha256

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
VECTOR_DIMENSIONS = 128
MAX_INDEX_CHUNKS = 100_000
MAX_MANIFEST_BYTES = 32 * 1024 * 1024


class RetrievalError(ValueError):
    """A retrieval configuration or index manifest is invalid."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RetrievalError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in TOKEN_RE.findall(text) if len(token) > 1)


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return 0.0 if not left_norm or not right_norm else dot / (left_norm * right_norm)


def _hashed_vector(tokens: Iterable[str], *, idf: dict[str, float] | None = None) -> dict[int, float]:
    vector: dict[int, float] = {}
    for token, count in Counter(tokens).items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSIONS
        sign = 1.0 if digest[4] & 1 else -1.0
        weight = float(count) * (idf.get(token, 1.0) if idf else 1.0)
        vector[index] = vector.get(index, 0.0) + sign * weight
    return vector


@dataclass(frozen=True)
class RankedChunk:
    chunk: Any
    score: float
    lexical_score: float
    vector_score: float


class RetrievalIndex:
    """A bounded in-memory index for transparent local strategies."""

    def __init__(self, chunks: Iterable[Any], *, strategy: str = "overlap", hybrid_weight: float = 0.5):
        bounded: list[Any] = []
        for chunk in chunks:
            if len(bounded) >= MAX_INDEX_CHUNKS:
                raise RetrievalError(f"index must contain 1..{MAX_INDEX_CHUNKS} chunks")
            bounded.append(chunk)
        self.chunks = tuple(bounded)
        if not self.chunks:
            raise RetrievalError(f"index must contain 1..{MAX_INDEX_CHUNKS} chunks")
        if strategy not in RETRIEVAL_STRATEGIES:
            raise RetrievalError(f"strategy must be one of {sorted(RETRIEVAL_STRATEGIES)}")
        if (
            isinstance(hybrid_weight, bool)
            or not isinstance(hybrid_weight, (int, float))
            or not 0 <= hybrid_weight <= 1
            or (isinstance(hybrid_weight, float) and not math.isfinite(hybrid_weight))
        ):
            raise RetrievalError("hybrid_weight must be a finite number between 0 and 1")
        identifiers = [chunk.chunk_id for chunk in self.chunks]
        if len(identifiers) != len(set(identifiers)):
            raise RetrievalError("chunk IDs must be unique")
        self.strategy = strategy
        self.hybrid_weight = float(hybrid_weight)
        self._tokens = {chunk.chunk_id: _tokenize(chunk.text) for chunk in self.chunks}
        self._counts = {identifier: Counter(tokens) for identifier, tokens in self._tokens.items()}
        document_frequency: Counter[str] = Counter()
        for tokens in self._tokens.values():
            document_frequency.update(set(tokens))
        count = len(self.chunks)
        self._bm25_idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self._tfidf_idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        self._average_length = sum(len(tokens) for tokens in self._tokens.values()) / count
        self._tfidf_vectors = {
            identifier: {token: frequency * self._tfidf_idf[token] for token, frequency in counts.items()}
            for identifier, counts in self._counts.items()
        }
        self._hash_vectors = {
            identifier: _hashed_vector(tokens, idf=self._tfidf_idf)
            for identifier, tokens in self._tokens.items()
        }

    @staticmethod
    def _mapping_cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(value * right.get(token, 0.0) for token, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return 0.0 if not left_norm or not right_norm else dot / (left_norm * right_norm)

    def _overlap(self, query_tokens: tuple[str, ...], chunk_id: str) -> float:
        query = set(query_tokens)
        return 0.0 if not query else len(query & set(self._tokens[chunk_id])) / len(query)

    def _bm25(self, query_tokens: tuple[str, ...], chunk_id: str) -> float:
        counts = self._counts[chunk_id]
        length = len(self._tokens[chunk_id])
        if not query_tokens or not length:
            return 0.0
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token in set(query_tokens):
            frequency = counts.get(token, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (1 - b + b * length / self._average_length)
            score += self._bm25_idf.get(token, 0.0) * (frequency * (k1 + 1)) / denominator
        return score

    def _tfidf(self, query_tokens: tuple[str, ...], chunk_id: str) -> float:
        query_counts = Counter(query_tokens)
        query = {token: frequency * self._tfidf_idf.get(token, 0.0) for token, frequency in query_counts.items()}
        return self._mapping_cosine(query, self._tfidf_vectors[chunk_id])

    def _hashing(self, query_tokens: tuple[str, ...], chunk_id: str) -> float:
        query = _hashed_vector(query_tokens, idf=self._tfidf_idf)
        return max(0.0, _cosine(query, self._hash_vectors[chunk_id]))

    def score(self, query: str, chunk: Any) -> RankedChunk:
        tokens = _tokenize(query)
        if self.strategy == "overlap":
            lexical = self._overlap(tokens, chunk.chunk_id)
            return RankedChunk(chunk, lexical, lexical, 0.0)
        if self.strategy == "bm25":
            lexical = self._bm25(tokens, chunk.chunk_id)
            return RankedChunk(chunk, lexical, lexical, 0.0)
        if self.strategy == "tfidf":
            lexical = self._tfidf(tokens, chunk.chunk_id)
            return RankedChunk(chunk, lexical, lexical, 0.0)
        lexical = self._bm25(tokens, chunk.chunk_id)
        vector = self._hashing(tokens, chunk.chunk_id)
        normalized_lexical = lexical / (1.0 + lexical)
        score = self.hybrid_weight * normalized_lexical + (1.0 - self.hybrid_weight) * vector
        return RankedChunk(chunk, score, lexical, vector)

    def rank(self, query: str, *, limit: int) -> tuple[RankedChunk, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise RetrievalError("limit must be an integer from 1 to 1000")
        ranked = sorted((self.score(query, chunk) for chunk in self.chunks), key=lambda row: (-row.score, row.chunk.chunk_id))
        return tuple(row for row in ranked[:limit] if row.score > 0)

    def manifest(self, *, suite_sha256: str, chunk_size: int, chunk_overlap: int) -> dict[str, Any]:
        chunks = [
            {
                "chunk_id": chunk.chunk_id,
                "source_id": chunk.source_id,
                "ordinal": chunk.ordinal,
                "document_sha256": chunk.document_sha256,
                "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                "token_count": len(self._tokens[chunk.chunk_id]),
            }
            for chunk in self.chunks
        ]
        unsigned = {
            "index_version": "1.0",
            "suite_sha256": suite_sha256,
            "strategy": self.strategy,
            "hybrid_weight": self.hybrid_weight,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
        return {**unsigned, "manifest_sha256": canonical_sha256(unsigned)}


def verify_manifest(manifest: Any) -> bool:
    if not isinstance(manifest, dict):
        return False
    expected = {
        "index_version", "suite_sha256", "strategy", "hybrid_weight", "chunk_size",
        "chunk_overlap", "chunk_count", "chunks", "manifest_sha256",
    }
    if set(manifest) != expected or manifest.get("index_version") != "1.0":
        return False
    if manifest.get("strategy") not in RETRIEVAL_STRATEGIES:
        return False
    suite_digest = manifest.get("suite_sha256")
    if not isinstance(suite_digest, str) or len(suite_digest) != 64 or any(char not in "0123456789abcdef" for char in suite_digest):
        return False
    weight = manifest.get("hybrid_weight")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not 0 <= weight <= 1
        or (isinstance(weight, float) and not math.isfinite(weight))
    ):
        return False
    size = manifest.get("chunk_size")
    overlap = manifest.get("chunk_overlap")
    count = manifest.get("chunk_count")
    if isinstance(size, bool) or not isinstance(size, int) or not 8 <= size <= 500:
        return False
    if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < size:
        return False
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= MAX_INDEX_CHUNKS:
        return False
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != count:
        return False
    seen: set[str] = set()
    digest_fields = ("document_sha256", "text_sha256")
    chunk_fields = {"chunk_id", "source_id", "ordinal", "document_sha256", "text_sha256", "token_count"}
    for chunk in chunks:
        if not isinstance(chunk, dict) or set(chunk) != chunk_fields:
            return False
        chunk_id = chunk.get("chunk_id")
        source_id = chunk.get("source_id")
        if not isinstance(chunk_id, str) or not chunk_id or len(chunk_id) > 512 or chunk_id in seen:
            return False
        if not isinstance(source_id, str) or not source_id or len(source_id) > 64:
            return False
        seen.add(chunk_id)
        if isinstance(chunk.get("ordinal"), bool) or not isinstance(chunk.get("ordinal"), int) or chunk["ordinal"] < 0:
            return False
        if isinstance(chunk.get("token_count"), bool) or not isinstance(chunk.get("token_count"), int) or chunk["token_count"] < 0:
            return False
        for field in digest_fields:
            digest = chunk.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                return False
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    return isinstance(claimed, str) and claimed == canonical_sha256(unsigned)


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    if not verify_manifest(manifest):
        raise RetrievalError("refusing to write an invalid index manifest")
    target = Path(path)
    if target.exists() and target.is_dir():
        raise RetrievalError("index manifest path cannot be a directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if len(encoded.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise RetrievalError(f"index manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_manifest(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        with target.open("rb") as stream:
            encoded = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise RetrievalError(f"index manifest exceeds {MAX_MANIFEST_BYTES} bytes")
        value = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
    except RetrievalError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RetrievalError(f"invalid index manifest: {exc}") from exc
    if not verify_manifest(value):
        raise RetrievalError("index manifest verification failed")
    return value
