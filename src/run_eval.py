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

from .evaluation import load_eval_set, run_mode


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
            mlflow.log_param("vector_backend", os.environ.get("VECTOR_BACKEND", "memory"))
            mlflow.log_metric("severity_accuracy", summary["severity_accuracy"])
            mlflow.log_metric("route_accuracy", summary["route_accuracy"])
            mlflow.log_metric("human_review_accuracy", summary["human_review_accuracy"])
            mlflow.log_metric("overall_accuracy", summary["overall_accuracy"])
            mlflow.log_metric("authoritative_citation_rate", summary["authoritative_citation_rate"])

            report_path = os.path.join(tempfile.gettempdir(), f"{mode}_eval_report.json")
            with open(report_path, "w") as f:
                json.dump(summary["cases"], f, indent=2)
            mlflow.log_artifact(report_path)

        print(f"\n=== Mode: {mode.upper()} ===")
        print(f"  Severity accuracy:          {summary['severity_accuracy']*100:.0f}%")
        print(f"  Route accuracy:             {summary['route_accuracy']*100:.0f}%")
        print(f"  Human review accuracy:      {summary['human_review_accuracy']*100:.0f}%")
        print(f"  Overall accuracy:           {summary['overall_accuracy']*100:.0f}%")
        print(f"  Authoritative citations:    {summary['authoritative_citation_rate']*100:.0f}%")
        for c in summary["cases"]:
            flag = "OK" if c["all_correct"] else "WRONG"
            print(
                f"    [{flag}] {c['id']} ({c['defect_type']}): "
                f"predicted={c['predicted_severity']} expected={c['expected_severity']} "
                f"cited={c['cited_source']}"
            )

    print("\n=== BEFORE / AFTER ===")
    print(f"  Buggy severity accuracy:  {results['buggy']['severity_accuracy']*100:.0f}%")
    print(f"  Fixed severity accuracy:  {results['fixed']['severity_accuracy']*100:.0f}%")
    print("\nRun `mlflow ui` in this directory to see both runs side by side.")


if __name__ == "__main__":
    main()
