"""
Runs the full eval set through the Defect Triage Agent in both "buggy" and
"fixed" mode, scores each run against ground truth, and logs everything to
MLflow — two tagged runs you can pull up side by side and compare.

This is the proof layer: instead of saying "I fixed the grounding bug and it
seems better," this script produces a real accuracy number, before and
after, that anyone can re-run and reproduce.

Usage:
    cd defect_triage_agent
    . venv/bin/activate
    python -m src.run_eval
"""
import json
import os
import tempfile
import mlflow

from . import graph as graph_module

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_eval_set():
    with open(os.path.join(_DATA_DIR, "eval_set.json"), "r") as f:
        return json.load(f)


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
    severity_acc = sum(c["severity_correct"] for c in scored_cases) / total
    route_acc = sum(c["route_correct"] for c in scored_cases) / total
    review_acc = sum(c["review_correct"] for c in scored_cases) / total
    overall_acc = sum(c["all_correct"] for c in scored_cases) / total

    return {
        "mode": mode,
        "n_cases": total,
        "severity_accuracy": severity_acc,
        "route_accuracy": route_acc,
        "human_review_accuracy": review_acc,
        "overall_accuracy": overall_acc,
        "cases": scored_cases,
    }


def main():
    eval_set = load_eval_set()
    mlflow.set_experiment("/defect-triage-agent-eval")

    results = {}
    for mode in ["buggy", "fixed"]:
        summary = run_mode(mode, eval_set)
        results[mode] = summary

        with mlflow.start_run(run_name=f"defect_triage_{mode}"):
            mlflow.set_tag("mode", mode)
            mlflow.log_param("n_eval_cases", summary["n_cases"])
            mlflow.log_metric("severity_accuracy", summary["severity_accuracy"])
            mlflow.log_metric("route_accuracy", summary["route_accuracy"])
            mlflow.log_metric("human_review_accuracy", summary["human_review_accuracy"])
            mlflow.log_metric("overall_accuracy", summary["overall_accuracy"])

            report_path = os.path.join(tempfile.gettempdir(), f"{mode}_eval_report.json")
            with open(report_path, "w") as f:
                json.dump(summary["cases"], f, indent=2)
            mlflow.log_artifact(report_path)

        print(f"\n=== Mode: {mode.upper()} ===")
        print(f"  Severity accuracy:      {summary['severity_accuracy']*100:.0f}%")
        print(f"  Route accuracy:         {summary['route_accuracy']*100:.0f}%")
        print(f"  Human review accuracy:  {summary['human_review_accuracy']*100:.0f}%")
        print(f"  Overall accuracy:       {summary['overall_accuracy']*100:.0f}%")
        for c in summary["cases"]:
            flag = "OK" if c["all_correct"] else "WRONG"
            print(
                f"    [{flag}] {c['id']} ({c['defect_type']}): "
                f"predicted={c['predicted_severity']} expected={c['expected_severity']} "
                f"cited={c['cited_source']}"
            )

    print("\n=== BEFORE / AFTER ===")
    print(
        f"  Buggy severity accuracy:  {results['buggy']['severity_accuracy']*100:.0f}%"
    )
    print(
        f"  Fixed severity accuracy:  {results['fixed']['severity_accuracy']*100:.0f}%"
    )
    print("\nRun `mlflow ui` in this directory to see both runs side by side.")


if __name__ == "__main__":
    main()
