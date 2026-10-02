"""Test setup: always run in deterministic mock mode with the in-memory store,
so results never depend on an API key, a model download, or a database."""
import os

os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ.pop("API_AUTH_TOKEN", None)
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["VECTOR_BACKEND"] = "memory"
os.environ["EMBEDDINGS_PROVIDER"] = "hashing"
