"""Bounded, dependency-free benchmark contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
import re
from typing import Any


ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
TRUST_STATES = {"trusted", "untrusted", "blocked"}
RETRIEVAL_STRATEGIES = {"overlap", "bm25", "tfidf", "hybrid"}
MAX_DOCUMENTS = 200
MAX_QUESTIONS = 500
MAX_CONTENT_CHARS = 200_000
MAX_TEXT_CHARS = 10_000
MAX_SUITE_BYTES = 5_000_000


class ContractError(ValueError):
    """Raised when public input violates the bounded schema."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{name} must be an object")
    return value


def _list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{name} must be an array")
    return value


def _string(value: Any, name: str, *, maximum: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    if len(value) > maximum:
        raise ContractError(f"{name} exceeds {maximum} characters")
    return value


def _identifier(value: Any, name: str) -> str:
    text = _string(value, name, maximum=64)
    if not ID_RE.fullmatch(text):
        raise ContractError(f"{name} has an invalid identifier")
    return text


def _iso_date(value: Any, name: str) -> date:
    text = _string(value, name, maximum=10)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ContractError(f"{name} must be an ISO date") from exc


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    source_id: str
    title: str
    source_url: str
    license: str
    observed_at: date
    trust: str
    content: str
    sha256: str
    expires_at: date | None = None

    @classmethod
    def from_dict(cls, raw: Any) -> "Document":
        data = _mapping(raw, "document")
        allowed = {
            "source_id", "title", "source_url", "license", "observed_at",
            "trust", "content", "sha256", "expires_at",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ContractError(f"document contains unknown fields: {sorted(unknown)}")
        content = _string(data.get("content"), "document.content", maximum=MAX_CONTENT_CHARS)
        digest = _string(data.get("sha256"), "document.sha256", maximum=64)
        if not SHA_RE.fullmatch(digest):
            raise ContractError("document.sha256 must be lowercase SHA-256")
        trust = _string(data.get("trust"), "document.trust", maximum=16)
        if trust not in TRUST_STATES:
            raise ContractError(f"document.trust must be one of {sorted(TRUST_STATES)}")
        expires = data.get("expires_at")
        return cls(
            source_id=_identifier(data.get("source_id"), "document.source_id"),
            title=_string(data.get("title"), "document.title", maximum=200),
            source_url=_string(data.get("source_url"), "document.source_url", maximum=500),
            license=_string(data.get("license"), "document.license", maximum=100),
            observed_at=_iso_date(data.get("observed_at"), "document.observed_at"),
            trust=trust,
            content=content,
            sha256=digest,
            expires_at=None if expires is None else _iso_date(expires, "document.expires_at"),
        )

    def validation_errors(self, evaluation_date: date) -> list[str]:
        errors: list[str] = []
        if content_sha256(self.content) != self.sha256:
            errors.append("hash_mismatch")
        if self.trust != "trusted":
            errors.append(f"trust_{self.trust}")
        if self.observed_at > evaluation_date:
            errors.append("observed_in_future")
        if self.expires_at is not None and self.expires_at < evaluation_date:
            errors.append("expired")
        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_url": self.source_url,
            "license": self.license,
            "observed_at": self.observed_at.isoformat(),
            "trust": self.trust,
            "content": self.content,
            "sha256": self.sha256,
            "expires_at": None if self.expires_at is None else self.expires_at.isoformat(),
        }


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[str, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "Claim":
        data = _mapping(raw, "claim")
        unknown = set(data) - {"text", "citations"}
        if unknown:
            raise ContractError(f"claim contains unknown fields: {sorted(unknown)}")
        citations = tuple(_identifier(item, "claim.citation") for item in _list(data.get("citations"), "claim.citations"))
        if len(citations) != len(set(citations)):
            raise ContractError("claim.citations contains duplicates")
        if not citations:
            raise ContractError("claim.citations must not be empty")
        return cls(text=_string(data.get("text"), "claim.text"), citations=citations)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "citations": list(self.citations)}


@dataclass(frozen=True)
class Response:
    answer: str | None
    claims: tuple[Claim, ...]

    @classmethod
    def from_dict(cls, raw: Any) -> "Response":
        data = _mapping(raw, "response")
        unknown = set(data) - {"answer", "claims"}
        if unknown:
            raise ContractError(f"response contains unknown fields: {sorted(unknown)}")
        answer = data.get("answer")
        if answer is not None:
            answer = _string(answer, "response.answer")
        claims = tuple(Claim.from_dict(item) for item in _list(data.get("claims", []), "response.claims"))
        if answer is None and claims:
            raise ContractError("an abstaining response cannot contain claims")
        if answer is not None and not claims:
            raise ContractError("an answered response must contain claims")
        return cls(answer=answer, claims=claims)

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.answer, "claims": [claim.to_dict() for claim in self.claims]}


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    expected_source_ids: tuple[str, ...]
    no_answer: bool
    adversarial: bool
    response: Response

    @classmethod
    def from_dict(cls, raw: Any) -> "Question":
        data = _mapping(raw, "question")
        allowed = {"question_id", "text", "expected_source_ids", "no_answer", "adversarial", "response"}
        unknown = set(data) - allowed
        if unknown:
            raise ContractError(f"question contains unknown fields: {sorted(unknown)}")
        expected = tuple(
            _identifier(item, "question.expected_source_id")
            for item in _list(data.get("expected_source_ids"), "question.expected_source_ids")
        )
        if len(expected) != len(set(expected)):
            raise ContractError("question.expected_source_ids contains duplicates")
        no_answer = data.get("no_answer")
        adversarial = data.get("adversarial", False)
        if not isinstance(no_answer, bool) or not isinstance(adversarial, bool):
            raise ContractError("question flags must be booleans")
        if no_answer and expected:
            raise ContractError("no-answer questions cannot expect sources")
        if not no_answer and not expected:
            raise ContractError("answerable questions must expect sources")
        response = Response.from_dict(data.get("response"))
        return cls(
            question_id=_identifier(data.get("question_id"), "question.question_id"),
            text=_string(data.get("text"), "question.text"),
            expected_source_ids=expected,
            no_answer=no_answer,
            adversarial=adversarial,
            response=response,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "expected_source_ids": list(self.expected_source_ids),
            "no_answer": self.no_answer,
            "adversarial": self.adversarial,
            "response": self.response.to_dict(),
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    schema_version: str
    suite_id: str
    version: str
    evaluation_date: date
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    documents: tuple[Document, ...]
    questions: tuple[Question, ...]
    retrieval_strategy: str = "overlap"
    hybrid_weight: float = 0.5

    @classmethod
    def from_dict(cls, raw: Any) -> "BenchmarkSuite":
        data = _mapping(raw, "suite")
        allowed = {
            "schema_version", "suite_id", "version", "evaluation_date", "chunk_size",
            "chunk_overlap", "retrieval_k", "documents", "questions",
            "retrieval_strategy", "hybrid_weight",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ContractError(f"suite contains unknown fields: {sorted(unknown)}")
        if data.get("schema_version") != "1.0":
            raise ContractError("suite.schema_version must be 1.0")
        version = _string(data.get("version"), "suite.version", maximum=32)
        if not SEMVER_RE.fullmatch(version):
            raise ContractError("suite.version must be semantic x.y.z")
        chunk_size = data.get("chunk_size")
        overlap = data.get("chunk_overlap")
        retrieval_k = data.get("retrieval_k")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or not 8 <= chunk_size <= 500:
            raise ContractError("suite.chunk_size must be between 8 and 500")
        if isinstance(overlap, bool) or not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
            raise ContractError("suite.chunk_overlap must be non-negative and smaller than chunk_size")
        if isinstance(retrieval_k, bool) or not isinstance(retrieval_k, int) or not 1 <= retrieval_k <= 20:
            raise ContractError("suite.retrieval_k must be between 1 and 20")
        strategy = data.get("retrieval_strategy", "overlap")
        if strategy not in RETRIEVAL_STRATEGIES:
            raise ContractError(f"suite.retrieval_strategy must be one of {sorted(RETRIEVAL_STRATEGIES)}")
        weight = data.get("hybrid_weight", 0.5)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not 0 <= weight <= 1
            or not math_is_finite(weight)
        ):
            raise ContractError("suite.hybrid_weight must be a finite number between 0 and 1")
        documents_raw = _list(data.get("documents"), "suite.documents")
        questions_raw = _list(data.get("questions"), "suite.questions")
        if not 1 <= len(documents_raw) <= MAX_DOCUMENTS:
            raise ContractError(f"suite.documents must contain 1..{MAX_DOCUMENTS} items")
        if not 1 <= len(questions_raw) <= MAX_QUESTIONS:
            raise ContractError(f"suite.questions must contain 1..{MAX_QUESTIONS} items")
        documents = tuple(Document.from_dict(item) for item in documents_raw)
        questions = tuple(Question.from_dict(item) for item in questions_raw)
        source_ids = [doc.source_id for doc in documents]
        question_ids = [question.question_id for question in questions]
        if len(source_ids) != len(set(source_ids)):
            raise ContractError("document source IDs must be unique")
        if len(question_ids) != len(set(question_ids)):
            raise ContractError("question IDs must be unique")
        unknown_sources = {
            source_id
            for question in questions
            for source_id in question.expected_source_ids
            if source_id not in set(source_ids)
        }
        if unknown_sources:
            raise ContractError(f"questions reference unknown expected sources: {sorted(unknown_sources)}")
        return cls(
            schema_version="1.0",
            suite_id=_identifier(data.get("suite_id"), "suite.suite_id"),
            version=version,
            evaluation_date=_iso_date(data.get("evaluation_date"), "suite.evaluation_date"),
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            retrieval_k=retrieval_k,
            documents=documents,
            questions=questions,
            retrieval_strategy=strategy,
            hybrid_weight=float(weight),
        )

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkSuite":
        if not isinstance(text, str):
            raise ContractError("suite JSON exceeds 5 MB")
        try:
            encoded_size = len(text.encode("utf-8"))
        except UnicodeEncodeError as exc:
            raise ContractError("suite must be valid UTF-8 text") from exc
        if encoded_size > MAX_SUITE_BYTES:
            raise ContractError("suite JSON exceeds 5 MB")
        try:
            raw = json.loads(text, object_pairs_hook=_unique_object)
        except ContractError:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            message = exc.msg if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ContractError(f"invalid JSON: {message}") from exc
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "version": self.version,
            "evaluation_date": self.evaluation_date.isoformat(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_k": self.retrieval_k,
            "documents": [document.to_dict() for document in self.documents],
            "questions": [question.to_dict() for question in self.questions],
        }
        # Preserve schema-1.0 digests for suites created before retrieval options
        # existed. Explicit defaults and omitted defaults are semantically equal.
        if self.retrieval_strategy != "overlap" or self.hybrid_weight != 0.5:
            value["retrieval_strategy"] = self.retrieval_strategy
            value["hybrid_weight"] = self.hybrid_weight
        return value

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


def math_is_finite(value: int | float) -> bool:
    """Avoid importing a numerical dependency for one contract check."""

    if isinstance(value, int):
        return True
    return value == value and value not in {float("inf"), float("-inf")}
