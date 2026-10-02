"""
HTTP service for the Defect Triage Agent.

Endpoints:
    GET  /healthz            liveness: the process is up
    GET  /readyz             readiness: the vector store is reachable and loaded
    POST /v1/triage          run the full graph, return the disposition
    POST /v1/triage/stream   same, streamed as server-sent events, one per agent

Security:
    - Set API_AUTH_TOKEN and every /v1 call must send it in the X-API-Key header
      (compared in constant time).
    - Report length is capped (MAX_REPORT_CHARS) to bound cost and abuse.
    - Secrets come from the environment (Kubernetes Secret in deployment); nothing
      sensitive is logged.

Run locally:
    uvicorn src.api:app --reload
"""
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Iterator, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import graph as graph_module
from . import llm
from .retrieval import get_retriever

MAX_REPORT_CHARS = int(os.environ.get("MAX_REPORT_CHARS", "4000"))

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("defect_triage.api")
logging.getLogger("httpx").setLevel(logging.WARNING)  # keep per-request client logs out of INFO


def _log(event: str, **fields) -> None:
    log.info(json.dumps({"event": event, **fields}))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class TriageRequest(BaseModel):
    raw_report: str = Field(..., min_length=10, max_length=MAX_REPORT_CHARS,
                            description="Free-text inspection note")
    mode: Literal["fixed", "buggy"] = Field("fixed", description="'buggy' reproduces the grounding bug")


class Citation(BaseModel):
    chunk_id: str
    source: str
    authoritative: bool


class TriageResponse(BaseModel):
    request_id: str
    mode: str
    defect_type: Optional[str]
    severity_tier: Optional[str]
    cited_source: Optional[str]
    citations: list[Citation]
    route: Optional[str]
    needs_human_review: Optional[bool]
    disposition_report: Optional[str]
    trace: list[str]
    latency_ms: float


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the compiled graph and the retriever (and embedding model) once, so
    # the first real request isn't the slow one. A failure here is logged, not
    # fatal: /readyz keeps the pod out of rotation until the store is usable.
    graph_module.get_graph()
    try:
        get_retriever()
    except Exception as exc:  # noqa: BLE001
        _log("retriever_init_failed", error=str(exc))
    yield


app = FastAPI(title="Defect Triage Agent", version="1.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    _log(
        "http_request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - start) * 1000, 1),
    )
    return response


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("API_AUTH_TOKEN")
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.exception_handler(llm.LLMOutputError)
async def llm_output_error(request: Request, exc: llm.LLMOutputError):
    _log("llm_output_error", request_id=getattr(request.state, "request_id", None), error=str(exc))
    return JSONResponse(status_code=502, content={"detail": "model returned an invalid response; try again"})


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    try:
        ready = get_retriever().is_ready()
    except Exception as exc:  # noqa: BLE001
        _log("readiness_check_failed", error=str(exc))
        ready = False
    if not ready:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready", "live_llm": llm.HAS_API_KEY}


def _to_response(result: dict, request_id: str, latency_ms: float) -> TriageResponse:
    return TriageResponse(
        request_id=request_id,
        mode=result["mode"],
        defect_type=result["defect_type"],
        severity_tier=result["severity_tier"],
        cited_source=result["cited_source"],
        citations=[Citation(**c) for c in result["citations"]],
        route=result["route"],
        needs_human_review=result["needs_human_review"],
        disposition_report=result["disposition_report"],
        trace=result["trace"],
        latency_ms=round(latency_ms, 1),
    )


@app.post("/v1/triage", response_model=TriageResponse, dependencies=[Depends(require_api_key)])
async def triage(body: TriageRequest, request: Request) -> TriageResponse:
    start = time.perf_counter()
    # The graph is synchronous; run it in a worker thread so the event loop
    # keeps serving other requests while the LLM calls are in flight.
    result = await run_in_threadpool(graph_module.run_triage, body.raw_report, body.mode)
    latency_ms = (time.perf_counter() - start) * 1000
    _log(
        "triage_complete",
        request_id=request.state.request_id,
        mode=body.mode,
        defect_type=result["defect_type"],
        severity_tier=result["severity_tier"],
        needs_human_review=result["needs_human_review"],
        latency_ms=round(latency_ms, 1),
    )
    return _to_response(result, request.state.request_id, latency_ms)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/triage/stream", dependencies=[Depends(require_api_key)])
def triage_stream(body: TriageRequest, request: Request) -> StreamingResponse:
    request_id = request.state.request_id

    def events() -> Iterator[str]:
        start = time.perf_counter()
        final_state: dict = {}
        try:
            for node, state in graph_module.stream_triage(body.raw_report, body.mode):
                final_state = state
                yield _sse(node, {
                    "node": node,
                    "step": state["trace"][-1] if state.get("trace") else None,
                    "defect_type": state.get("defect_type"),
                    "severity_tier": state.get("severity_tier"),
                    "route": state.get("route"),
                    "needs_human_review": state.get("needs_human_review"),
                })
            latency_ms = (time.perf_counter() - start) * 1000
            yield _sse("done", _to_response(final_state, request_id, latency_ms).model_dump())
        except llm.LLMOutputError:
            yield _sse("error", {"detail": "model returned an invalid response; try again"})

    # A sync generator: Starlette iterates it in a threadpool, so streaming
    # doesn't block the event loop.
    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "x-request-id": request_id},
    )
