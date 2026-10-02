import math

import pytest

from src.retrieval import (
    Chunk,
    HashingEmbedder,
    Hit,
    InMemoryVectorStore,
    build_memory_retriever,
    chunk_text,
    load_knowledge_chunks,
    rerank,
)


def test_hashing_embedder_is_deterministic_and_unit_length():
    emb = HashingEmbedder()
    a1, a2 = emb.embed(["hairline crack on the leading edge"] * 2)
    assert a1 == a2
    assert len(a1) == emb.dim == 384
    assert math.isclose(math.sqrt(sum(x * x for x in a1)), 1.0, rel_tol=1e-9)


def test_chunk_text_keeps_short_text_whole():
    assert chunk_text("A short note.") == ["A short note."]


def test_chunk_text_splits_long_text_with_overlap():
    text = " ".join(f"Sentence number {i} about turbine blade inspection." for i in range(60))
    chunks = chunk_text(text, max_chars=300, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # consecutive chunks share context
    assert chunks[0][-30:].split()[-1] in chunks[1]


def test_knowledge_chunks_cover_every_defect_in_both_sources():
    chunks = load_knowledge_chunks()
    policy = {c.defect_type for c in chunks if c.source == "severity_policy"}
    guide = {c.defect_type for c in chunks if c.source == "reference_guide"}
    assert policy == guide and len(policy) == 6
    assert all(c.authoritative for c in chunks if c.source == "severity_policy")
    assert not any(c.authoritative for c in chunks if c.source == "reference_guide")
    assert all("severity_tier" in c.payload for c in chunks if c.source == "severity_policy")


@pytest.fixture(scope="module")
def retriever():
    return build_memory_retriever(HashingEmbedder())


def test_authoritative_filter_keeps_the_outdated_guide_out(retriever):
    hits = retriever.retrieve("hairline crack on the blade", k=10, authoritative_only=True)
    assert hits and all(h.chunk.authoritative for h in hits)


def test_unfiltered_retrieval_returns_both_sources_for_a_defect(retriever):
    hits = retriever.retrieve("hairline crack on the blade", k=4, defect_type="surface_crack")
    assert {h.chunk.source for h in hits} == {"severity_policy", "reference_guide"}


def test_defect_type_filter_is_exact(retriever):
    hits = retriever.retrieve("anything", k=10, defect_type="porosity")
    assert hits and all(h.chunk.defect_type == "porosity" for h in hits)


def test_rerank_breaks_vector_ties_by_term_coverage():
    def hit(cid, content):
        chunk = Chunk(cid, "severity_policy", True, "x", "v1", content)
        return Hit(chunk=chunk, score=0.5, vector_score=0.5)

    ranked = rerank("thermal discoloration near root", [hit("a", "unrelated text"), hit("b", "thermal discoloration at the root")])
    assert ranked[0].chunk.chunk_id == "b"
    assert ranked[0].lexical_score > ranked[1].lexical_score


def test_in_memory_store_reports_ready_only_when_loaded():
    store = InMemoryVectorStore()
    assert not store.is_ready()
    store.upsert([Chunk("c", "severity_policy", True, "x", "v1", "text")], [[1.0] + [0.0] * 383])
    assert store.is_ready()
