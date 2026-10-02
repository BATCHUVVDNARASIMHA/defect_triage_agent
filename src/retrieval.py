"""
Retrieval layer for the risk-assessment agent.

Both knowledge sources, the authoritative severity policy and the outdated
reference guide, are chunked, embedded, and stored in a vector store with
metadata: source, an authoritative flag, defect_type, and document version.

The grounding fix now has two layers:

  1. Prompt: the fixed-mode prompt tells the model the policy is authoritative.
  2. Retrieval: fixed mode filters on authoritative=True, so the outdated guide
     never reaches the model's context in the first place.

Buggy mode retrieves without the filter, which reproduces the original failure:
both sources land in context with nothing telling the model which one wins.

Backends (VECTOR_BACKEND):
  memory    default; in-process store built from data/ at startup, no database
  pgvector  Postgres + pgvector via DATABASE_URL; load it with `python -m src.ingest`

Embeddings (EMBEDDINGS_PROVIDER):
  hashing    default; deterministic feature hashing, offline, used in tests and CI
  fastembed  BAAI/bge-small-en-v1.5 (384-dim) via fastembed, used in deployment
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Iterable, Optional, Protocol, Sequence

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

EMBEDDING_DIM = 384
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Chunks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str            # "severity_policy" or "reference_guide"
    authoritative: bool
    defect_type: str
    doc_version: str
    content: str
    payload: dict = field(default_factory=dict)  # structured fields from the source entry


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    vector_score: float
    lexical_score: float = 0.0


def chunk_text(text: str, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Split text into chunks of at most max_chars, breaking on sentence ends
    where possible and carrying `overlap` characters of context forward."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_end = text.rfind(". ", start + max_chars // 2, end)
            if sentence_end != -1:
                end = sentence_end + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _load_json(filename: str) -> dict:
    with open(os.path.join(_DATA_DIR, filename), "r") as f:
        return json.load(f)


def load_knowledge_chunks() -> list[Chunk]:
    """Turn both JSON sources into retrieval chunks with metadata."""
    chunks: list[Chunk] = []

    policy = _load_json("severity_policy.json")
    policy_version = str(policy.get("_last_updated", "unknown"))
    for defect_type, entry in policy["defect_types"].items():
        text = (
            f"Defect type: {defect_type}. Severity tier: {entry['severity_tier']}. "
            f"Route: {entry['route']}. Human review required: {entry['requires_human_review']}. "
            f"Rationale: {entry.get('rationale', '')}"
        )
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(
                chunk_id=f"severity_policy:{defect_type}:{i}",
                source="severity_policy",
                authoritative=True,
                defect_type=defect_type,
                doc_version=policy_version,
                content=piece,
                payload=dict(entry),
            ))

    guide = _load_json("reference_guide.json")
    guide_version = str(guide.get("_last_updated", "unknown"))
    for defect_type, note in guide["notes"].items():
        text = f"Defect type: {defect_type}. Reference note: {note}"
        for i, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(
                chunk_id=f"reference_guide:{defect_type}:{i}",
                source="reference_guide",
                authoritative=False,
                defect_type=defect_type,
                doc_version=guide_version,
                content=piece,
                payload={"note": note},
            ))

    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams).

    Not semantic, but stable across runs and machines, needs no network, and is
    good enough for metadata-filtered retrieval over a small corpus. Used in
    tests and CI so results never depend on a model download.
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        toks = _tokens(text)
        features = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for feat in features:
            digest = hashlib.md5(feat.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 via fastembed (ONNX, no torch). 384 dimensions."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", cache_dir: Optional[str] = None):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = EMBEDDING_DIM

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._model.embed(list(texts))]


def get_embedder() -> Embedder:
    provider = os.environ.get("EMBEDDINGS_PROVIDER", "hashing").lower()
    if provider == "fastembed":
        return FastEmbedEmbedder(cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
    if provider == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDINGS_PROVIDER: {provider!r}")


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------
# Vector stores
# ---------------------------------------------------------------------------
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        k: int,
        authoritative_only: bool = False,
        defect_type: Optional[str] = None,
    ) -> list[Hit]: ...

    def is_ready(self) -> bool: ...


class InMemoryVectorStore:
    def __init__(self):
        self._rows: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._rows[chunk.chunk_id] = (chunk, list(vec))

    def search(self, vector, k, authoritative_only=False, defect_type=None) -> list[Hit]:
        hits = []
        for chunk, vec in self._rows.values():
            if authoritative_only and not chunk.authoritative:
                continue
            if defect_type is not None and chunk.defect_type != defect_type:
                continue
            score = _cosine(vector, vec)
            hits.append(Hit(chunk=chunk, score=score, vector_score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def is_ready(self) -> bool:
        return bool(self._rows)


def _vector_literal(vec: Iterable[float]) -> str:
    return "[" + ",".join(f"{float(x):.7f}" for x in vec) + "]"


class PgVectorStore:
    """Postgres + pgvector. Metadata lives in plain columns so filters are exact
    SQL predicates, and an HNSW index serves the cosine-distance ordering."""

    TABLE = "triage_chunks"

    def __init__(self, dsn: str, dim: int = EMBEDDING_DIM):
        self._dsn = dsn
        self._dim = int(dim)

    def _connect(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True, connect_timeout=5)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    chunk_id      TEXT PRIMARY KEY,
                    source        TEXT NOT NULL,
                    authoritative BOOLEAN NOT NULL,
                    defect_type   TEXT NOT NULL,
                    doc_version   TEXT,
                    content       TEXT NOT NULL,
                    payload       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding     vector({self._dim}) NOT NULL
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.TABLE}_embedding_hnsw "
                f"ON {self.TABLE} USING hnsw (embedding vector_cosine_ops)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.TABLE}_metadata "
                f"ON {self.TABLE} (defect_type, authoritative)"
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        rows = [
            (
                c.chunk_id, c.source, c.authoritative, c.defect_type, c.doc_version,
                c.content, json.dumps(c.payload), _vector_literal(v),
            )
            for c, v in zip(chunks, vectors)
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {self.TABLE}
                        (chunk_id, source, authoritative, defect_type, doc_version, content, payload, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        source = EXCLUDED.source,
                        authoritative = EXCLUDED.authoritative,
                        defect_type = EXCLUDED.defect_type,
                        doc_version = EXCLUDED.doc_version,
                        content = EXCLUDED.content,
                        payload = EXCLUDED.payload,
                        embedding = EXCLUDED.embedding
                    """,
                    rows,
                )

    def search(self, vector, k, authoritative_only=False, defect_type=None) -> list[Hit]:
        query = f"""
            SELECT chunk_id, source, authoritative, defect_type, doc_version, content, payload,
                   1 - (embedding <=> %(vec)s::vector) AS score
            FROM {self.TABLE}
            WHERE (NOT %(auth_only)s OR authoritative)
              AND (%(defect)s::text IS NULL OR defect_type = %(defect)s::text)
            ORDER BY embedding <=> %(vec)s::vector
            LIMIT %(k)s
        """
        params = {
            "vec": _vector_literal(vector),
            "auth_only": bool(authoritative_only),
            "defect": defect_type,
            "k": int(k),
        }
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        hits = []
        for chunk_id, source, auth, dtype, version, content, payload, score in rows:
            if isinstance(payload, str):
                payload = json.loads(payload)
            chunk = Chunk(chunk_id, source, auth, dtype, version, content, payload or {})
            hits.append(Hit(chunk=chunk, score=float(score), vector_score=float(score)))
        return hits

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute(f"SELECT count(*) FROM {self.TABLE}").fetchone()[0])

    def is_ready(self) -> bool:
        """Ready means reachable AND loaded, so a pod doesn't take traffic before ingestion."""
        try:
            return self.count() > 0
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Retriever with lexical reranking
# ---------------------------------------------------------------------------
def rerank(query: str, hits: Sequence[Hit], lexical_weight: float = 0.3) -> list[Hit]:
    """Hybrid rerank: blend vector similarity with query-term coverage.

    Vector search finds the right neighbourhood; term coverage breaks ties in
    favour of chunks that literally mention what the inspector wrote.
    """
    q_terms = set(_tokens(query))
    reranked = []
    for h in hits:
        d_terms = set(_tokens(h.chunk.content))
        lexical = len(q_terms & d_terms) / len(q_terms) if q_terms else 0.0
        score = (1 - lexical_weight) * h.vector_score + lexical_weight * lexical
        reranked.append(replace(h, score=score, lexical_score=lexical))
    reranked.sort(key=lambda h: h.score, reverse=True)
    return reranked


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, use_rerank: bool = True):
        self.store = store
        self.embedder = embedder
        self.use_rerank = use_rerank

    def retrieve(
        self,
        query: str,
        k: int = 4,
        authoritative_only: bool = False,
        defect_type: Optional[str] = None,
        candidates: int = 10,
    ) -> list[Hit]:
        vector = self.embedder.embed([query])[0]
        hits = self.store.search(
            vector, k=max(k, candidates), authoritative_only=authoritative_only, defect_type=defect_type
        )
        if self.use_rerank:
            hits = rerank(query, hits)
        return hits[:k]

    def is_ready(self) -> bool:
        return self.store.is_ready()


def build_memory_retriever(embedder: Optional[Embedder] = None) -> Retriever:
    embedder = embedder or get_embedder()
    store = InMemoryVectorStore()
    chunks = load_knowledge_chunks()
    store.upsert(chunks, embedder.embed([c.content for c in chunks]))
    return Retriever(store, embedder)


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    backend = os.environ.get("VECTOR_BACKEND", "memory").lower()
    if backend == "pgvector":
        embedder = get_embedder()
        return Retriever(PgVectorStore(os.environ["DATABASE_URL"], dim=embedder.dim), embedder)
    if backend == "memory":
        return build_memory_retriever()
    raise ValueError(f"Unknown VECTOR_BACKEND: {backend!r}")
