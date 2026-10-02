"""
Release gate for CI: runs the eval set in fixed mode and fails the build if
any metric drops below its threshold. A prompt, retrieval, or model change
that reintroduces the grounding bug can't merge.

Usage:
    python -m src.eval_gate
"""
import sys

from .evaluation import load_eval_set, run_mode

THRESHOLDS = {
    "severity_accuracy": 1.0,
    "route_accuracy": 1.0,
    "human_review_accuracy": 1.0,
    "authoritative_citation_rate": 1.0,
}


def main() -> int:
    eval_set = load_eval_set()
    fixed = run_mode("fixed", eval_set)
    buggy = run_mode("buggy", eval_set)

    failures = []
    print(f"Release gate on {fixed['n_cases']} cases (fixed mode):")
    for metric, threshold in THRESHOLDS.items():
        value = fixed[metric]
        ok = value >= threshold
        print(f"  {'PASS' if ok else 'FAIL'}  {metric:<30} {value:.0%}  (threshold {threshold:.0%})")
        if not ok:
            failures.append(metric)

    print(f"\nBuggy mode severity accuracy (should stay low): {buggy['severity_accuracy']:.0%}")

    if failures:
        print(f"\nGate failed on: {', '.join(failures)}")
        return 1
    print("\nGate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
