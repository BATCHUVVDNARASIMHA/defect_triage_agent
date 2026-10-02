# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

# Fixed numeric UID so Kubernetes runAsNonRoot can verify the user.
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

COPY src ./src
COPY data ./data

# Bake the embedding model into the image so pods start without reaching Hugging Face.
RUN python -c "from src.retrieval import FastEmbedEmbedder; FastEmbedEmbedder(cache_dir='${FASTEMBED_CACHE_PATH}').embed(['warm up'])" \
 && chown -R app:app /app

ENV HF_HUB_OFFLINE=1 \
    VECTOR_BACKEND=memory \
    EMBEDDINGS_PROVIDER=fastembed \
    UVICORN_WORKERS=1

USER 10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status == 200 else 1)"

# One worker per container by default; scale with replicas, not workers.
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS} --proxy-headers"]
