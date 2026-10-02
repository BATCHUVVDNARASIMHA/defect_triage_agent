"""
Thin LLM wrapper.

If ANTHROPIC_API_KEY is set in the environment, agents call the real Claude
API. If not, agents fall back to a deterministic mock so the whole pipeline
still runs end to end and still demonstrates the grounding bug/fix — useful
for development, testing, and demoing without burning API credits.

To run in LIVE mode:  export ANTHROPIC_API_KEY=sk-ant-...

Reliability:
  - the SDK retries transient API errors (rate limits, 5xx, timeouts) with
    exponential backoff; ANTHROPIC_MAX_RETRIES and ANTHROPIC_TIMEOUT_S tune it
  - call_claude_json() validates structured output and re-asks once if the
    model returns something unparseable, then raises LLMOutputError instead of
    letting a malformed answer flow downstream
"""
import json
import os
import re
from typing import Callable, Optional

from . import observability

HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

_client = None


class LLMOutputError(RuntimeError):
    """The model's output could not be parsed or failed validation."""


def _get_client():
    global _client
    if _client is None:
        import anthropic

        client = anthropic.Anthropic(
            max_retries=int(os.environ.get("ANTHROPIC_MAX_RETRIES", "3")),
            timeout=float(os.environ.get("ANTHROPIC_TIMEOUT_S", "30")),
        )
        _client = observability.wrap_anthropic_client(client)
    return _client


def call_claude(system_prompt: str, user_prompt: str) -> str:
    """Real API call. Only used when HAS_API_KEY is True."""
    response = _get_client().messages.create(
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


def call_claude_json(
    system_prompt: str,
    user_prompt: str,
    validate: Optional[Callable[[dict], None]] = None,
    attempts: int = 2,
) -> dict:
    """Call Claude for a JSON answer, validate it, and re-ask once on failure."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        prompt = user_prompt
        if attempt > 0:
            prompt += (
                "\n\nYour previous reply was not a valid JSON object in the required "
                "format. Reply with ONLY the JSON object."
            )
        text = call_claude(system_prompt, prompt)
        try:
            data = extract_json(text)
            if validate is not None:
                validate(data)
            return data
        except (ValueError, KeyError, TypeError) as exc:  # json.JSONDecodeError is a ValueError
            last_error = exc
    raise LLMOutputError(f"Model output failed validation after {attempts} attempts: {last_error}")
