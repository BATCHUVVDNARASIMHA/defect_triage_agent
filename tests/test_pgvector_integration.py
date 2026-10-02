"""Runs against a real Postgres + pgvector (CI starts one as a service).
Skipped when TEST_DATABASE_URL isn't set."""
import os

import pytest

from src.retrieval import HashingEmbedder, PgVectorStore, Retriever, load_knowledge_chunks

DSN = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DSN, reason="TEST_DATABASE_URL not set"),
]


@pytest.fixture(scope="module")
def store():
    s = PgVectorStore(DSN)
    s.ensure_schema()
    emb = HashingEmbedder()
    chunks = load_knowledge_chunks()
    s.upsert(chunks, emb.embed([c.content for c in chunks]))
    return s


def test_ingest_is_idempotent(store):
    before = store.count()
    emb = HashingEmbedder()
    chunks = load_knowledge_chunks()
    store.upsert(chunks, emb.embed([c.content for c in chunks]))
    assert store.count() == before == len(chunks)
    assert store.is_ready()


def test_pgvector_filters_match_the_in_memory_store(store):
    retriever = Retriever(store, HashingEmbedder())
    auth = retriever.retrieve("hairline crack on the blade", k=10, authoritative_only=True)
    assert auth and all(h.chunk.authoritative for h in auth)

    both = retriever.retrieve("hairline crack on the blade", k=4, defect_type="surface_crack")
    assert {h.chunk.source for h in both} == {"severity_policy", "reference_guide"}
    policy_hit = next(h for h in both if h.chunk.authoritative)
    assert policy_hit.chunk.payload["severity_tier"] == "CRITICAL"
