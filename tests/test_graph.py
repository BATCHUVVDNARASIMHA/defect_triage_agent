from src import graph
from src.evaluation import load_eval_set, run_mode


def test_fixed_mode_passes_the_full_eval_set():
    summary = run_mode("fixed", load_eval_set())
    assert summary["severity_accuracy"] == 1.0
    assert summary["route_accuracy"] == 1.0
    assert summary["human_review_accuracy"] == 1.0
    assert summary["authoritative_citation_rate"] == 1.0


def test_buggy_mode_still_reproduces_the_grounding_bug():
    summary = run_mode("buggy", load_eval_set())
    assert summary["severity_accuracy"] == 0.0
    assert summary["authoritative_citation_rate"] == 0.0


def test_fixed_mode_never_puts_the_reference_guide_in_context():
    for case in load_eval_set():
        result = graph.run_triage(case["raw_report"], mode="fixed")
        assert result["citations"], case["id"]
        assert all(c["authoritative"] for c in result["citations"]), case["id"]


def test_review_required_cases_reach_the_human_gate():
    for case in load_eval_set():
        result = graph.run_triage(case["raw_report"], mode="fixed")
        escalated = any(step.startswith("[human_escalation]") for step in result["trace"])
        assert escalated == case["expected_human_review"], case["id"]


def test_unrecognized_report_goes_to_manual_review_without_a_guessed_severity():
    result = graph.run_triage("The mounting bracket looks a bit odd today.", mode="fixed")
    assert result["defect_type"] is None
    assert result["severity_tier"] is None
    assert result["route"] == "MANUAL_REVIEW"
    assert result["needs_human_review"] is True


def test_stream_emits_each_agent_in_order():
    nodes = [name for name, _ in graph.stream_triage("Hairline crack on the leading edge, 4mm.", "fixed")]
    assert nodes == ["intake_agent", "risk_assessment_agent", "routing_agent", "report_agent", "human_escalation"]


def test_invalid_mode_is_rejected():
    try:
        graph.initial_state("Hairline crack on the leading edge.", mode="yolo")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
