# Defect Triage Agent

A multi-agent LangGraph system that triages manufacturing quality-inspection
reports, deliberately built to reproduce and fix the same class of grounding
bug found in production RAG/agent systems — where an AI trusts a
plausible-sounding but outdated document over the actual authoritative
source.

Built and tested August 2026.

## Architecture

```
raw inspection note
        |
        v
  [intake_agent]            -- parses free text into a structured defect_type
        |
        v
  [risk_assessment_agent]   -- determines severity tier
        |                       (this is where the grounding bug lives)
        v
  [routing_agent]           -- determines disposition route + review flag
        |
        v
  [report_agent]            -- drafts the inspector-facing report
        |
        v
  conditional edge
        |
   +----+----+
   |         |
[escalate] [finish]
   |         |
   v         v
[human_    [END]
escalation]
   |
   v
 [END]
```

Five specialized agents, a stateful graph, tool calls, and a conditional
edge that routes critical or review-required cases to a human gate instead
of auto-finalizing. Built with LangGraph + LangChain.

## The grounding bug (the point of the project)

`risk_assessment_agent` has access to two sources when it looks up a
defect's severity:

- **`data/severity_policy.json`** — the authoritative policy. Always
  correct, always current.
- **`data/reference_guide.json`** — a plausible-sounding new-hire reference
  doc that reads naturally but is out of date and disagrees with the policy
  on every single defect type.

In **buggy mode**, the agent is given both sources with no instruction
about which to trust. In **fixed mode**, the agent is explicitly instructed
that the policy tool is always authoritative and the reference guide must
be ignored when the two disagree.

This is the same shape of failure as an AI customer-support agent citing a
marketing document's warranty claim instead of the actual return-policy
tool — except this version, including the data, the graph, and the eval
harness, was built from scratch rather than following a workshop template.

## Results (reproducible — see below)

Ran the full 6-case eval set in both modes, scored against ground truth,
logged to MLflow:

| Mode  | Severity accuracy | Route accuracy | Human-review accuracy |
|-------|-------------------|-----------------|------------------------|
| Buggy | 0%                 | 100%            | 100%                   |
| Fixed | 100%               | 100%            | 100%                   |

The buggy mode isn't just "worse" — it's wrong on every single severity
call, because the reference guide's phrasing consistently contradicts the
real policy (it downplays a genuinely critical surface crack as "cosmetic,"
and overcorrects a minor edge nick to "always scrap"). Route and human-review
accuracy hold at 100% in both modes on this eval set — the routing and
escalation logic downstream isn't what's broken, the severity call feeding
into it is. That's deliberate: a project that shows a small improvement is
a weaker proof than one that shows the failure mode is real and the fix is
complete.

## Running it yourself

```bash
cd defect_triage_agent
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Mock mode (no API key needed) — runs immediately, deterministic:
python -m src.run_eval

# Live mode — set your key first, agents make real Claude API calls:
export ANTHROPIC_API_KEY=sk-ant-...
python -m src.run_eval

# View results side by side:
mlflow ui
```

## Project structure

```
defect_triage_agent/
  data/
    severity_policy.json    # authoritative source
    reference_guide.json    # the outdated "trap" source
    eval_set.json           # 6 test cases with ground truth
  src/
    tools.py                # get_severity_policy(), get_reference_guide()
    llm.py                  # Claude API wrapper + mock fallback
    graph.py                # the LangGraph state machine, 5 agents
    run_eval.py             # buggy vs. fixed eval, MLflow logging
  notebooks/
    databricks_deploy_notebook.py   # Databricks-ready version, Unity
                                     # Catalog + CI/CD notes
  requirements.txt
  README.md
```

## Notes on framing this honestly

This is a **personal project**, built and dated August 2026, independent of
any employer or coursework. It is not part of the Honeywell Aerospace
capstone — the domain (manufacturing defect triage) is thematically
adjacent to that project's subject matter, but the two are unrelated
systems built at different times for different purposes, and should be
described as such if asked.

Suggested resume line:

> **Defect Triage Agent (Personal Project)** — Aug 2026
> Built a 5-agent LangGraph workflow for manufacturing defect triage with
> conditional human-escalation routing. Reproduced and fixed a real
> grounding defect where the agent trusted an outdated reference document
> over the authoritative policy tool, proving the fix with an MLflow-logged
> eval showing severity-classification accuracy improve from 0% to 100%
> across a 6-case test set.

This gives you a fully ownable, fully defensible answer to "walk me through
your LangGraph/agentic workflow experience" — one you built yourself, that
you can open the code for, that you can re-run live in an interview if
asked.
