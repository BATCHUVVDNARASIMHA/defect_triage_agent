"""
Load both knowledge sources into Postgres + pgvector.

Idempotent (upserts by chunk_id), so it is safe to run on every deploy. The
Helm chart runs it as a post-install/post-upgrade hook and docker-compose runs
it once before the API starts.

Usage:
    DATABASE_URL=postgresql://... EMBEDDINGS_PROVIDER=fastembed python -m src.ingest
"""
import os
import sys
import time

from .retrieval import PgVectorStore, get_embedder, load_knowledge_chunks


def _wait_for_database(store: PgVectorStore, attempts: int = 30, delay_s: float = 2.0) -> None:
    for attempt in range(1, attempts + 1):
        try:
            with store._connect() as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:  # noqa: BLE001 - surface any connection failure
            print(f"database not ready (attempt {attempt}/{attempts}): {exc}", file=sys.stderr)
            time.sleep(delay_s)
    raise SystemExit("database never became reachable")


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is not set")

    embedder = get_embedder()
    store = PgVectorStore(dsn, dim=embedder.dim)
    _wait_for_database(store)
    store.ensure_schema()

    chunks = load_knowledge_chunks()
    vectors = embedder.embed([c.content for c in chunks])
    store.upsert(chunks, vectors)

    authoritative = sum(c.authoritative for c in chunks)
    print(
        f"ingested {len(chunks)} chunks ({authoritative} authoritative, "
        f"{len(chunks) - authoritative} non-authoritative); table now holds {store.count()} rows"
    )


if __name__ == "__main__":
    main()
