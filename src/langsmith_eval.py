"""
Runs the eval set as LangSmith experiments, one per mode, so buggy and fixed
show up side by side in the LangSmith UI with per-example scores and the full
trace behind every answer (retrieved chunks, prompts, tokens, latency).

Requires:
    LANGSMITH_API_KEY      your LangSmith key
    LANGSMITH_TRACING=true so the agent runs are traced into the experiment
Optional:
    ANTHROPIC_API_KEY      live Claude calls (without it the deterministic mock runs)
    LANGSMITH_DATASET      dataset name (default: defect-triage-eval)

Usage:
    python -m src.langsmith_eval
"""
import os
import sys

from . import graph as graph_module
from .evaluation import cited_authoritative, load_eval_set

DATASET_NAME = os.environ.get("LANGSMITH_DATASET", "defect-triage-eval")


def ensure_dataset(client) -> str:
    """Create the dataset from data/eval_set.json the first time; reuse it after."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        return DATASET_NAME
    dataset = client.create_dataset(
        DATASET_NAME,
        description="Defect triage eval: inspection reports with ground-truth severity, route, and review flag.",
    )
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {
                "inputs": {"raw_report": case["raw_report"]},
                "outputs": {
                    "defect_type": case["expected_defect_type"],
                    "severity_tier": case["expected_severity_tier"],
                    "route": case["expected_route"],
                    "needs_human_review": case["expected_human_review"],
                },
                "metadata": {"case_id": case["id"], "note": case.get("note", "")},
            }
            for case in load_eval_set()
        ],
    )
    return DATASET_NAME


# Evaluators: each compares the agent's output to the reference output.
def severity_correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["severity_tier"] == reference_outputs["severity_tier"]


def route_correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["route"] == reference_outputs["route"]


def review_correct(outputs: dict, reference_outputs: dict) -> bool:
    return outputs["needs_human_review"] == reference_outputs["needs_human_review"]


def cites_authoritative_source(outputs: dict) -> bool:
    return cited_authoritative(outputs)


def make_target(mode: str):
    def target(inputs: dict) -> dict:
        result = graph_module.run_triage(inputs["raw_report"], mode=mode)
        return {
            "defect_type": result["defect_type"],
            "severity_tier": result["severity_tier"],
            "route": result["route"],
            "needs_human_review": result["needs_human_review"],
            "cited_source": result["cited_source"],
            "citations": result["citations"],
        }

    return target


def main() -> int:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("LANGSMITH_API_KEY is not set; nothing to run.", file=sys.stderr)
        return 1

    from langsmith import Client, evaluate

    client = Client()
    dataset = ensure_dataset(client)
    for mode in ("buggy", "fixed"):
        results = evaluate(
            make_target(mode),
            data=dataset,
            evaluators=[severity_correct, route_correct, review_correct, cites_authoritative_source],
            experiment_prefix=f"defect-triage-{mode}",
            metadata={
                "mode": mode,
                "live_llm": bool(os.environ.get("ANTHROPIC_API_KEY")),
                "vector_backend": os.environ.get("VECTOR_BACKEND", "memory"),
            },
            max_concurrency=2,
        )
        print(f"{mode}: experiment {results.experiment_name}")
    print("Open the dataset in LangSmith to compare the two experiments side by side.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
