"""
LangSmith tracing helpers.

Tracing turns on when LANGSMITH_TRACING=true and LANGSMITH_API_KEY is set.
LangGraph node runs are traced automatically in that case; the helpers here
add the two things LangGraph can't see on its own:

  - the raw Anthropic API calls (wrapped client: prompts, outputs, tokens, latency)
  - the retrieval step (traced as a "retriever" run, so LangSmith shows exactly
    which chunks reached the model's context)

If the langsmith package isn't installed, everything degrades to a no-op so the
agent still runs.
"""
import os

try:
    from langsmith import traceable as _traceable
    from langsmith.wrappers import wrap_anthropic as _wrap_anthropic

    _HAS_LANGSMITH = True
except ImportError:  # pragma: no cover - langsmith is in requirements
    _HAS_LANGSMITH = False


def tracing_enabled() -> bool:
    return (
        os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        and bool(os.environ.get("LANGSMITH_API_KEY"))
    )


def traceable(*args, **kwargs):
    """langsmith.traceable when available, otherwise a pass-through decorator."""
    if _HAS_LANGSMITH:
        return _traceable(*args, **kwargs)
    if args and callable(args[0]) and not kwargs:
        return args[0]

    def decorator(func):
        return func

    return decorator


def wrap_anthropic_client(client):
    """Wrap the Anthropic client so every call shows up in the LangSmith trace."""
    if _HAS_LANGSMITH and tracing_enabled():
        return _wrap_anthropic(client)
    return client
