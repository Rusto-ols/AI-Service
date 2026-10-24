import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import storage  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)

VALID = {
    "farm_id": "FARM-001", "region": "Krasnodar", "crop_type": "wheat", "area_ha": 2500,
    "temperature_avg": 24.3, "precipitation_mm": 320, "payment_delay_days": 45,
    "previous_defaults": 1, "debt": 6500000,
}


@pytest.fixture(autouse=True)
def clean_storage():
    storage.clear()
    yield


def test_01_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_02_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200 and r.json()["model_version"] == "1.0"


def test_03_predict_ok():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 201
    assert r.json()["risk_score"] == 0.9 and r.json()["risk_level"] == "high"


def test_04_negative_area():
    r = client.post("/predict", json={**VALID, "area_ha": -100})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "area_ha"]


def test_05_unknown_region():
    r = client.post("/predict", json={**VALID, "region": "Moscow"})
    assert r.status_code == 400 and r.json() == {"detail": "Unknown region"}


def test_06_existing_id():
    rid = client.post("/predict", json=VALID).json()["request_id"]
    assert client.get(f"/predictions/{rid}").status_code == 200


def test_07_unknown_id():
    r = client.get("/predictions/abc-123")
    assert r.status_code == 404 and r.json() == {"detail": "Prediction not found"}


def test_08_limit():
    for _ in range(3):
        client.post("/predict", json=VALID)
    r = client.get("/predictions?limit=2")
    assert r.status_code == 200 and len(r.json()) == 2


def test_09_filter_high():
    client.post("/predict", json=VALID)
    client.post("/predict", json={**VALID, "payment_delay_days": 0, "previous_defaults": 0, "debt": 1000})
    r = client.get("/predictions?risk_level=high")
    assert r.status_code == 200 and all(x["risk_level"] == "high" for x in r.json())
    assert len(r.json()) == 1


def test_10_negative_limit():
    assert client.get("/predictions?limit=-5").status_code == 422
