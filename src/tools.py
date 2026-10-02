"""
Tools available to the agents.

There are deliberately TWO sources of information here, exactly mirroring the
grounding defect from the Databricks Agent Apps Workshop:

  - get_severity_policy()  -> the AUTHORITATIVE source (the policy tool)
  - get_reference_guide()  -> a plausible-sounding but OUTDATED source
                               (the marketing-doc equivalent)

The whole point of this project is to test whether the agent correctly trusts
the policy over the reference guide when the two disagree.

search_knowledge() is the retrieval tool the risk-assessment agent uses: it
searches both sources in the vector store and can filter to authoritative
chunks only. The direct lookups stay for deterministic business rules
(routing) and for the intake agent's list of valid defect types.
"""
import json
import os

from .observability import traceable
from .retrieval import get_retriever

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _load(filename: str) -> dict:
    with open(os.path.join(_DATA_DIR, filename), "r") as f:
        return json.load(f)


def get_severity_policy(defect_type: str) -> dict:
    """The authoritative tool. Always correct, always current."""
    policy = _load("severity_policy.json")
    entry = policy["defect_types"].get(defect_type)
    if entry is None:
        return {"error": f"No policy entry found for defect_type='{defect_type}'"}
    return {"source": "severity_policy.json (AUTHORITATIVE)", **entry}


def get_reference_guide(defect_type: str) -> dict:
    """The trap. Reads naturally, sounds helpful, is out of date."""
    guide = _load("reference_guide.json")
    note = guide["notes"].get(defect_type)
    if note is None:
        return {"error": f"No reference note found for defect_type='{defect_type}'"}
    return {"source": "reference_guide.json (NOT AUTHORITATIVE - may be outdated)", "note": note}


def get_all_defect_types() -> list:
    policy = _load("severity_policy.json")
    return list(policy["defect_types"].keys())


@traceable(run_type="retriever", name="search_knowledge")
def search_knowledge(
    query: str,
    defect_type: str | None = None,
    authoritative_only: bool = False,
    k: int = 4,
) -> list[dict]:
    """Vector search over both sources. Returns LangSmith-style documents so the
    trace shows exactly which chunks reached the model, with their metadata."""
    hits = get_retriever().retrieve(
        query, k=k, authoritative_only=authoritative_only, defect_type=defect_type
    )
    return [
        {
            "type": "Document",
            "page_content": h.chunk.content,
            "metadata": {
                "chunk_id": h.chunk.chunk_id,
                "source": h.chunk.source,
                "authoritative": h.chunk.authoritative,
                "defect_type": h.chunk.defect_type,
                "doc_version": h.chunk.doc_version,
                "score": round(h.score, 4),
                "payload": h.chunk.payload,
            },
        }
        for h in hits
    ]
