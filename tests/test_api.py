import pytest
from fastapi.testclient import TestClient

from src.api import app

CRACK = "Inspector found a hairline crack along the leading edge, approximately 4mm long."


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan warm-up
        yield c


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"


def test_triage_returns_grounded_disposition(client):
    resp = client.post("/v1/triage", json={"raw_report": CRACK, "mode": "fixed"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["severity_tier"] == "CRITICAL"
    assert body["route"] == "SCRAP"
    assert body["needs_human_review"] is True
    assert body["citations"] and all(c["authoritative"] for c in body["citations"])
    assert resp.headers["x-request-id"] == body["request_id"]


def test_buggy_mode_is_available_for_demos(client):
    body = client.post("/v1/triage", json={"raw_report": CRACK, "mode": "buggy"}).json()
    assert body["severity_tier"] == "MINOR"  # the grounding bug, reproduced on purpose
    assert body["cited_source"].startswith("reference_guide")


@pytest.mark.parametrize(
    "payload",
    [
        {"raw_report": "short"},
        {"raw_report": "x" * 5000},
        {"raw_report": CRACK, "mode": "yolo"},
        {},
    ],
)
def test_bad_input_is_rejected(client, payload):
    assert client.post("/v1/triage", json=payload).status_code == 422


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_AUTH_TOKEN", "s3cret")
    assert client.post("/v1/triage", json={"raw_report": CRACK}).status_code == 401
    assert client.post("/v1/triage", json={"raw_report": CRACK}, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = client.post("/v1/triage", json={"raw_report": CRACK}, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200


def test_stream_emits_agent_events_then_done(client):
    with client.stream("POST", "/v1/triage/stream", json={"raw_report": CRACK}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())
    events = [line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("event: ")]
    assert events == [
        "intake_agent", "risk_assessment_agent", "routing_agent",
        "report_agent", "human_escalation", "done",
    ]
