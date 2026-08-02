"""
Defect Triage Agent — a LangGraph multi-agent workflow.

    intake_agent -> risk_assessment_agent -> routing_agent -> report_agent -> (conditional) human_escalation / END

Each node is a specialized agent with one job. State flows through all of
them. The routing after report_agent is CONDITIONAL: critical or
review-required cases get flagged for a human instead of auto-finalizing.

The interesting node is risk_assessment_agent: it has access to two sources
that can disagree (the policy tool vs the reference guide), and the `mode`
flag controls whether its instructions tell it which one to trust. This is
what lets run_eval.py demonstrate a real, measurable before/after fix.
"""
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from . import tools
from . import llm

# ---------------------------------------------------------------------------
# Deterministic mock behavior — used when no API key is present.
# BUGGY_MOCK_MAP simulates what an ungrounded model tends to infer from the
# reference guide's casual phrasing. FIXED mode always reads the policy tool.
# ---------------------------------------------------------------------------
BUGGY_MOCK_SEVERITY = {
    "surface_crack": "MINOR",
    "coating_discoloration": "MAJOR",
    "dimensional_deviation": "MINOR",
    "porosity": "MINOR",
    "edge_nick": "CRITICAL",
    "thermal_discoloration": "MINOR",
}

_KEYWORD_MAP = [
    (["crack"], "surface_crack"),
    (["thermal", "heat", "bluish"], "thermal_discoloration"),
    (["coating"], "coating_discoloration"),
    (["thickness", "tolerance", "midspan"], "dimensional_deviation"),
    (["porosity", "void", "x-ray"], "porosity"),
    (["nick"], "edge_nick"),
]


class TriageState(TypedDict):
    raw_report: str
    mode: str                      # "buggy" or "fixed"
    defect_type: Optional[str]
    severity_tier: Optional[str]
    cited_source: Optional[str]
    route: Optional[str]
    needs_human_review: Optional[bool]
    disposition_report: Optional[str]
    trace: list


# ---------------------------------------------------------------------------
# Node 1: Intake agent
# ---------------------------------------------------------------------------
def intake_agent(state: TriageState) -> TriageState:
    raw = state["raw_report"].lower()

    if llm.HAS_API_KEY:
        system = (
            "You extract structured defect data from raw inspection notes. "
            f"Valid defect_type values are: {tools.get_all_defect_types()}. "
            'Respond ONLY with JSON: {"defect_type": "<one of the valid values>"}'
        )
        raw_response = llm.call_claude(system, state["raw_report"])
        parsed = llm.extract_json(raw_response)
        defect_type = parsed["defect_type"]
    else:
        defect_type = None
        for keywords, dtype in _KEYWORD_MAP:
            if any(k in raw for k in keywords):
                defect_type = dtype
                break

    state["defect_type"] = defect_type
    state["trace"].append(f"[intake_agent] parsed defect_type = {defect_type}")
    return state


# ---------------------------------------------------------------------------
# Node 2: Risk assessment agent  <-- this is where the grounding bug lives
# ---------------------------------------------------------------------------
def risk_assessment_agent(state: TriageState) -> TriageState:
    defect_type = state["defect_type"]
    policy = tools.get_severity_policy(defect_type)
    reference = tools.get_reference_guide(defect_type)

    if llm.HAS_API_KEY:
        if state["mode"] == "fixed":
            system = (
                "You are a manufacturing quality risk assessor. You will be given "
                "two sources of information about a defect type: a policy tool "
                "and a reference guide. THE POLICY TOOL IS ALWAYS AUTHORITATIVE. "
                "If the reference guide disagrees with the policy tool, ignore the "
                "reference guide entirely and use the policy tool's severity_tier. "
                'Respond ONLY with JSON: {"severity_tier": "...", "cited_source": "..."}'
            )
        else:
            # BUGGY: no priority instruction given, both sources presented as
            # equally valid. This reproduces the exact class of failure from
            # the Databricks workshop.
            system = (
                "You are a manufacturing quality risk assessor. Determine the "
                "severity_tier (CRITICAL, MAJOR, or MINOR) for this defect using "
                "the information provided below. "
                'Respond ONLY with JSON: {"severity_tier": "...", "cited_source": "..."}'
            )

        user = f"Policy tool output: {policy}\nReference guide output: {reference}"
        raw_response = llm.call_claude(system, user)
        parsed = llm.extract_json(raw_response)
        severity_tier = parsed["severity_tier"]
        cited_source = parsed.get("cited_source", "unknown")
    else:
        if state["mode"] == "fixed":
            severity_tier = policy["severity_tier"]
            cited_source = "severity_policy.json"
        else:
            severity_tier = BUGGY_MOCK_SEVERITY[defect_type]
            cited_source = "reference_guide.json"

    state["severity_tier"] = severity_tier
    state["cited_source"] = cited_source
    state["trace"].append(
        f"[risk_assessment_agent:{state['mode']}] severity_tier = {severity_tier} "
        f"(cited: {cited_source})"
    )
    return state


# ---------------------------------------------------------------------------
# Node 3: Routing agent — deterministic business rule off the POLICY tool.
# Routing always uses the authoritative policy regardless of mode, because in
# a real plant, routing/disposition follows a fixed rulebook per defect type.
# The bug we're testing lives in interpretation (risk tier), not in the
# published routing rules.
# ---------------------------------------------------------------------------
def routing_agent(state: TriageState) -> TriageState:
    policy = tools.get_severity_policy(state["defect_type"])
    state["route"] = policy["route"]
    state["needs_human_review"] = policy["requires_human_review"]
    state["trace"].append(
        f"[routing_agent] route = {state['route']}, "
        f"needs_human_review = {state['needs_human_review']}"
    )
    return state


# ---------------------------------------------------------------------------
# Node 4: Report agent
# ---------------------------------------------------------------------------
def report_agent(state: TriageState) -> TriageState:
    if llm.HAS_API_KEY:
        system = (
            "Write a short, plain-English disposition report for a quality "
            "inspector. State the defect, the severity tier, the route, and "
            "whether human review is required. Two sentences maximum."
        )
        user = (
            f"Defect type: {state['defect_type']}, severity: {state['severity_tier']}, "
            f"route: {state['route']}, human review required: {state['needs_human_review']}"
        )
        report = llm.call_claude(system, user)
    else:
        report = (
            f"Defect classified as {state['defect_type']} "
            f"({state['severity_tier']} severity). "
            f"Disposition: {state['route']}. "
            f"Human review required: {state['needs_human_review']}."
        )

    state["disposition_report"] = report.strip()
    state["trace"].append("[report_agent] disposition report drafted")
    return state


# ---------------------------------------------------------------------------
# Node 5: Human escalation gate (only reached conditionally)
# ---------------------------------------------------------------------------
def human_escalation(state: TriageState) -> TriageState:
    state["trace"].append(
        "[human_escalation] FLAGGED — routed to human inspector for sign-off, "
        "not auto-finalized"
    )
    return state


def _needs_escalation(state: TriageState) -> str:
    return "escalate" if state["needs_human_review"] else "finish"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(TriageState)

    graph.add_node("intake_agent", intake_agent)
    graph.add_node("risk_assessment_agent", risk_assessment_agent)
    graph.add_node("routing_agent", routing_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("human_escalation", human_escalation)

    graph.set_entry_point("intake_agent")
    graph.add_edge("intake_agent", "risk_assessment_agent")
    graph.add_edge("risk_assessment_agent", "routing_agent")
    graph.add_edge("routing_agent", "report_agent")
    graph.add_conditional_edges(
        "report_agent",
        _needs_escalation,
        {"escalate": "human_escalation", "finish": END},
    )
    graph.add_edge("human_escalation", END)

    return graph.compile()


def run_triage(raw_report: str, mode: str = "fixed") -> TriageState:
    app = build_graph()
    initial_state: TriageState = {
        "raw_report": raw_report,
        "mode": mode,
        "defect_type": None,
        "severity_tier": None,
        "cited_source": None,
        "route": None,
        "needs_human_review": None,
        "disposition_report": None,
        "trace": [],
    }
    return app.invoke(initial_state)
