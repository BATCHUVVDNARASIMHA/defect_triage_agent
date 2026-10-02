"""
Defect Triage Agent — a LangGraph multi-agent workflow.

    intake_agent -> risk_assessment_agent -> routing_agent -> report_agent -> (conditional) human_escalation / END
         |
         +--> unrecognized_defect -> human_escalation   (when the report can't be classified)

Each node is a specialized agent with one job. State flows through all of
them. The routing after report_agent is CONDITIONAL: critical or
review-required cases get flagged for a human instead of auto-finalizing.

The interesting node is risk_assessment_agent: it retrieves evidence from a
vector store holding two sources that can disagree (the authoritative policy vs
the outdated reference guide). The `mode` flag controls both whether retrieval
filters to authoritative chunks and whether the prompt says which source to
trust. That is what lets the eval demonstrate a real, measurable before/after.
"""
from functools import lru_cache
from typing import Iterator, Optional, TypedDict

from langgraph.graph import END, StateGraph

from . import llm, tools

VALID_SEVERITIES = {"CRITICAL", "MAJOR", "MINOR"}

# ---------------------------------------------------------------------------
# Deterministic mock behavior — used when no API key is present.
# BUGGY_MOCK_SEVERITY simulates what an ungrounded model tends to infer from the
# reference guide's casual phrasing. FIXED mode always reads the policy chunk.
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
    citations: list                # chunks that were in the model's context
    route: Optional[str]
    needs_human_review: Optional[bool]
    disposition_report: Optional[str]
    trace: list


# ---------------------------------------------------------------------------
# Node 1: Intake agent
# ---------------------------------------------------------------------------
def intake_agent(state: TriageState) -> TriageState:
    raw = state["raw_report"].lower()
    valid_types = tools.get_all_defect_types()

    if llm.HAS_API_KEY:
        system = (
            "You extract structured defect data from raw inspection notes. "
            f"Valid defect_type values are: {valid_types}. If none fit, use \"unknown\". "
            'Respond ONLY with JSON: {"defect_type": "<one of the valid values or unknown>"}'
        )

        def _validate(data: dict) -> None:
            if data["defect_type"] not in valid_types + ["unknown"]:
                raise ValueError(f"invalid defect_type {data['defect_type']!r}")

        parsed = llm.call_claude_json(system, state["raw_report"], validate=_validate)
        defect_type = parsed["defect_type"] if parsed["defect_type"] in valid_types else None
    else:
        defect_type = None
        for keywords, dtype in _KEYWORD_MAP:
            if any(k in raw for k in keywords):
                defect_type = dtype
                break

    state["defect_type"] = defect_type
    state["trace"].append(f"[intake_agent] parsed defect_type = {defect_type}")
    return state


def _after_intake(state: TriageState) -> str:
    return "assess" if state["defect_type"] else "unrecognized"


# ---------------------------------------------------------------------------
# Node 1b: Unrecognized defect — never guess a severity for something we
# couldn't classify; send it straight to a person.
# ---------------------------------------------------------------------------
def unrecognized_defect(state: TriageState) -> TriageState:
    state["route"] = "MANUAL_REVIEW"
    state["needs_human_review"] = True
    state["disposition_report"] = (
        "The report could not be matched to a known defect type, so no severity was "
        "assigned. Routed to an inspector for manual classification."
    )
    state["trace"].append("[unrecognized_defect] no known defect type; skipping automated assessment")
    return state


# ---------------------------------------------------------------------------
# Node 2: Risk assessment agent  <-- this is where the grounding bug lives
# ---------------------------------------------------------------------------
def _format_context(docs: list[dict]) -> str:
    return "\n".join(
        f"[{d['metadata']['chunk_id']}] (source: {d['metadata']['source']}, "
        f"version {d['metadata']['doc_version']}) {d['page_content']}"
        for d in docs
    )


def risk_assessment_agent(state: TriageState) -> TriageState:
    defect_type = state["defect_type"]
    fixed = state["mode"] == "fixed"

    # Fixed mode filters retrieval to authoritative chunks; buggy mode doesn't,
    # so the outdated reference guide lands in context next to the policy.
    docs = tools.search_knowledge(
        state["raw_report"], defect_type=defect_type, authoritative_only=fixed, k=4
    )
    citations = [
        {
            "chunk_id": d["metadata"]["chunk_id"],
            "source": d["metadata"]["source"],
            "authoritative": d["metadata"]["authoritative"],
        }
        for d in docs
    ]
    chunk_ids = [c["chunk_id"] for c in citations]

    if llm.HAS_API_KEY:
        if fixed:
            system = (
                "You are a manufacturing quality risk assessor. You will be given "
                "retrieved context chunks. The severity_policy source is ALWAYS "
                "AUTHORITATIVE. If any other source disagrees with it, ignore that "
                "source entirely and use the policy's severity tier. Use only the "
                "context provided, and cite the chunk id you relied on. "
                'Respond ONLY with JSON: {"severity_tier": "...", "cited_source": "<chunk id>"}'
            )
        else:
            # BUGGY: no priority instruction, both sources presented as equally
            # valid. Reproduces the exact class of failure from the Databricks workshop.
            system = (
                "You are a manufacturing quality risk assessor. Determine the "
                "severity_tier (CRITICAL, MAJOR, or MINOR) for this defect using "
                "the context provided below, and cite the chunk id you relied on. "
                'Respond ONLY with JSON: {"severity_tier": "...", "cited_source": "<chunk id>"}'
            )

        def _validate(data: dict) -> None:
            if data["severity_tier"] not in VALID_SEVERITIES:
                raise ValueError(f"invalid severity_tier {data['severity_tier']!r}")

        user = (
            f"Inspection report: {state['raw_report']}\n"
            f"Defect type: {defect_type}\n\nRetrieved context:\n{_format_context(docs)}"
        )
        parsed = llm.call_claude_json(system, user, validate=_validate)
        severity_tier = parsed["severity_tier"]
        cited_source = parsed.get("cited_source") or "unknown"
    else:
        if fixed:
            policy_doc = next(d for d in docs if d["metadata"]["authoritative"])
            severity_tier = policy_doc["metadata"]["payload"]["severity_tier"]
            cited_source = policy_doc["metadata"]["chunk_id"]
        else:
            severity_tier = BUGGY_MOCK_SEVERITY[defect_type]
            guide_doc = next((d for d in docs if not d["metadata"]["authoritative"]), docs[0])
            cited_source = guide_doc["metadata"]["chunk_id"]

    state["severity_tier"] = severity_tier
    state["cited_source"] = cited_source
    state["citations"] = citations
    state["trace"].append(
        f"[risk_assessment_agent:{state['mode']}] severity_tier = {severity_tier} "
        f"(cited: {cited_source}; retrieved: {', '.join(chunk_ids)})"
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
            "inspector. State the defect, the severity tier, the route, whether "
            "human review is required, and the source id the severity came from. "
            "Two sentences maximum."
        )
        user = (
            f"Defect type: {state['defect_type']}, severity: {state['severity_tier']}, "
            f"route: {state['route']}, human review required: {state['needs_human_review']}, "
            f"source: {state['cited_source']}"
        )
        report = llm.call_claude(system, user)
    else:
        report = (
            f"Defect classified as {state['defect_type']} "
            f"({state['severity_tier']} severity, source: {state['cited_source']}). "
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
    graph.add_node("unrecognized_defect", unrecognized_defect)
    graph.add_node("risk_assessment_agent", risk_assessment_agent)
    graph.add_node("routing_agent", routing_agent)
    graph.add_node("report_agent", report_agent)
    graph.add_node("human_escalation", human_escalation)

    graph.set_entry_point("intake_agent")
    graph.add_conditional_edges(
        "intake_agent",
        _after_intake,
        {"assess": "risk_assessment_agent", "unrecognized": "unrecognized_defect"},
    )
    graph.add_edge("unrecognized_defect", "human_escalation")
    graph.add_edge("risk_assessment_agent", "routing_agent")
    graph.add_edge("routing_agent", "report_agent")
    graph.add_conditional_edges(
        "report_agent",
        _needs_escalation,
        {"escalate": "human_escalation", "finish": END},
    )
    graph.add_edge("human_escalation", END)

    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    """Compile once per process; the API reuses it across requests."""
    return build_graph()


def initial_state(raw_report: str, mode: str = "fixed") -> TriageState:
    if mode not in ("fixed", "buggy"):
        raise ValueError(f"mode must be 'fixed' or 'buggy', got {mode!r}")
    return {
        "raw_report": raw_report,
        "mode": mode,
        "defect_type": None,
        "severity_tier": None,
        "cited_source": None,
        "citations": [],
        "route": None,
        "needs_human_review": None,
        "disposition_report": None,
        "trace": [],
    }


def run_triage(raw_report: str, mode: str = "fixed") -> TriageState:
    return get_graph().invoke(
        initial_state(raw_report, mode),
        config={"run_name": f"defect_triage_{mode}", "tags": [f"mode:{mode}"]},
    )


def stream_triage(raw_report: str, mode: str = "fixed") -> Iterator[tuple[str, dict]]:
    """Yield (node_name, state_after_node) as each agent finishes."""
    for update in get_graph().stream(
        initial_state(raw_report, mode),
        config={"run_name": f"defect_triage_{mode}", "tags": [f"mode:{mode}", "stream"]},
        stream_mode="updates",
    ):
        for node_name, node_state in update.items():
            yield node_name, node_state
