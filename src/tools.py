"""
Tools available to the risk-assessment agent.

There are deliberately TWO sources of information here, exactly mirroring the
grounding defect from the Databricks Agent Apps Workshop:

  - get_severity_policy()  -> the AUTHORITATIVE source (the policy tool)
  - get_reference_guide()  -> a plausible-sounding but OUTDATED source
                               (the marketing-doc equivalent)

The whole point of this project is to test whether the agent's prompt
correctly instructs it to trust the policy tool over the reference guide
when the two disagree.
"""
import json
import os

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
