"""
Thin LLM wrapper.

If ANTHROPIC_API_KEY is set in the environment, agents call the real Claude
API. If not, agents fall back to a deterministic mock so the whole pipeline
still runs end to end and still demonstrates the grounding bug/fix — useful
for development, testing, and demoing without burning API credits.

To run in LIVE mode:  export ANTHROPIC_API_KEY=sk-ant-...
"""
import os
import json
import re

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

if HAS_API_KEY:
    import anthropic
    _client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"


def call_claude(system_prompt: str, user_prompt: str) -> str:
    """Real API call. Only used when HAS_API_KEY is True."""
    response = _client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def extract_json(text: str) -> dict:
    """Pull the first {...} JSON object out of a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text!r}")
    return json.loads(match.group(0))
