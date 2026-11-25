
import asyncio
from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

import config
from main import app

VALID = {"farm_id": "FARM-001", "region": "Krasnodar", "crop_type": "wheat", "area_ha": 2500,
         "temperature_avg": 24.3, "precipitation_mm": 320, "payment_delay_days": 45,
         "previous_defaults": 1, "debt": 6500000}


def llm_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"model": "mock-model", "choices": [
        {"message": {"role": "assistant", "content": "Объяснение риска."}, "finish_reason": "stop"}]})


def use_llm(client: TestClient, handler) -> None:
    client.app.state.llm.client = httpx.AsyncClient(base_url="https://llm.test",
                                                    transport=httpx.MockTransport(handler))


@pytest.fixture()
def client():
    with TestClient(app) as c:
        use_llm(c, llm_ok)
        yield c


def test_01_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and "X-Request-ID" in r.headers and "X-Process-Time" in r.headers


def test_02_predict_ok_and_request_id_passthrough(client):
    r = client.post("/predict", json=VALID, headers={"X-Request-ID": "my-trace-001"})
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == "my-trace-001" == r.headers["X-Request-ID"]
    assert body["risk_score"] == 0.9 and body["risk_level"] == "high" and body["source"] == "single"


def test_03_negative_area(client):
    r = client.post("/predict", json={**VALID, "area_ha": -100})
    assert r.status_code == 422 and r.json()["request_id"] == r.headers["X-Request-ID"]


def test_04_unknown_region(client):
    r = client.post("/predict", json={**VALID, "region": "Moscow"})
    assert r.status_code == 400 and "Unknown region" in r.json()["detail"]
    assert r.json()["request_id"]


def test_05_existing_id_from_sqlite(client):
    rid = client.post("/predict", json=VALID).json()["request_id"]
    r = client.get(f"/predictions/{rid}")
    assert r.status_code == 200 and r.json()["request_id"] == rid


def test_06_unknown_id(client):
    assert client.get("/predictions/abc-123").status_code == 404


def test_07_batch_predict_and_409(client):
    items = [{**VALID, "farm_id": f"FARM-00{i}"} for i in range(3)]
    r = client.post("/batch-predict", json={"items": items})
    assert r.status_code == 200 and r.json()["count"] == 3
    rid = r.json()["request_id"]
    assert client.get(f"/predictions/{rid}").status_code == 409
    batch = client.get(f"/requests/{rid}/predictions")
    assert batch.status_code == 200 and len(batch.json()) == 3


def test_08_batch_too_large(client):
    items = [VALID] * (config.MAX_BATCH_SIZE + 1)
    assert client.post("/batch-predict", json={"items": items}).status_code == 422


def test_09_predict_queued(client):
    r = client.post("/predict-queued", json=VALID)
    assert r.status_code == 200 and r.json()["source"] == "queued"


def test_10_queue_full(client):
    batcher = client.app.state.batcher
    full = asyncio.Queue(maxsize=1)
    full.put_nowait(object())
    original, batcher.queue = batcher.queue, full
    try:
        r = client.post("/predict-queued", json=VALID)
    finally:
        batcher.queue = original
    assert r.status_code == 429 and r.headers["Retry-After"] == "1"


def test_11_by_weekday(client):
    client.post("/predict", json=VALID)
    today = datetime.now(timezone.utc).strftime("%A").lower()
    r = client.get(f"/predictions/by-weekday?weekday={today}&page=1&per_page=5")
    assert r.status_code == 200
    meta = r.json()["meta"]
    assert meta["total_items"] >= 1 and meta["current_page"] == 1 and meta["prev_page"] is None


def test_12_negative_page(client):
    assert client.get("/predictions/by-weekday?weekday=monday&page=-1").status_code == 422


def test_13_unknown_weekday(client):
    assert client.get("/predictions/by-weekday?weekday=funday").status_code == 400


def test_14_llm_chat(client):
    r = client.post("/llm/chat", json={"prompt": "Что такое агроскоринг?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Объяснение риска." and body["returned_model"] == "mock-model"
    assert body["requested_model"] == "auto" and body["finish_reason"] == "stop"


@pytest.mark.parametrize("handler, code", [
    (lambda req: httpx.Response(400, json={"error": "bad"}), 502),
    (lambda req: httpx.Response(200, json={"unexpected": True}), 502),
    (lambda req: httpx.Response(503), 503),
    (lambda req: (_ for _ in ()).throw(httpx.ConnectError("down")), 503),
    (lambda req: (_ for _ in ()).throw(httpx.ReadTimeout("slow")), 504),
])
def test_15_llm_errors(client, handler, code):
    use_llm(client, handler)
    r = client.post("/llm/chat", json={"prompt": "test"})
    assert r.status_code == code and r.json()["request_id"]


def test_16_predict_with_llm(client):
    r = client.post("/predict-with-llm", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"]["risk_score"] == 0.9 and body["prediction"]["risk_level"] == "high"
    assert body["explanation"] == "Объяснение риска."


def test_17_inference_timeout(client, monkeypatch):
    monkeypatch.setattr(config, "INFERENCE_TIMEOUT_SECONDS", 0.001)
    r = client.post("/predict", json=VALID)
    assert r.status_code == 504 and r.json()["detail"] == "Model inference timed out"


def test_18_queue_status(client):
    r = client.get("/queue-status")
    assert r.status_code == 200 and r.json()["max_batch_size"] == config.MAX_BATCH_SIZE
