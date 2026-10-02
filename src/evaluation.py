"""
Scoring shared by the MLflow eval (run_eval.py), the LangSmith eval
(langsmith_eval.py), and the CI release gate (eval_gate.py).
"""
import json
import os

from . import graph as graph_module

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_eval_set() -> list:
    with open(os.path.join(_DATA_DIR, "eval_set.json"), "r") as f:
        return json.load(f)


def cited_authoritative(result: dict) -> bool:
    """True when the source the severity call cites is an authoritative chunk."""
    cited = result.get("cited_source") or ""
    for c in result.get("citations") or []:
        if c["chunk_id"] == cited:
            return bool(c["authoritative"])
    return cited.startswith("severity_policy")


def score_case(case: dict, result: dict) -> dict:
    severity_correct = result["severity_tier"] == case["expected_severity_tier"]
    route_correct = result["route"] == case["expected_route"]
    review_correct = result["needs_human_review"] == case["expected_human_review"]
    return {
        "id": case["id"],
        "defect_type": case["expected_defect_type"],
        "severity_correct": severity_correct,
        "route_correct": route_correct,
        "review_correct": review_correct,
        "all_correct": severity_correct and route_correct and review_correct,
        "cited_authoritative": cited_authoritative(result),
        "predicted_severity": result["severity_tier"],
        "expected_severity": case["expected_severity_tier"],
        "cited_source": result["cited_source"],
    }


def run_mode(mode: str, eval_set: list) -> dict:
    scored_cases = []
    for case in eval_set:
        result = graph_module.run_triage(case["raw_report"], mode=mode)
        scored_cases.append(score_case(case, result))

    total = len(scored_cases)
    return {
        "mode": mode,
        "n_cases": total,
        "severity_accuracy": sum(c["severity_correct"] for c in scored_cases) / total,
        "route_accuracy": sum(c["route_correct"] for c in scored_cases) / total,
        "human_review_accuracy": sum(c["review_correct"] for c in scored_cases) / total,
        "overall_accuracy": sum(c["all_correct"] for c in scored_cases) / total,
        "authoritative_citation_rate": sum(c["cited_authoritative"] for c in scored_cases) / total,
        "cases": scored_cases,
    }
