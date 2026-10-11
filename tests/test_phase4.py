"""Phase 4 Verification Tests.

Validates:
  1. Constraint #3: Hard schema rejection (HTTP 422) on missing/wrong columns.
  2. Alert aggregation logic: Port scan burst yields ONE alert with hit_count > 1.
  3. Lazy SHAP explanation: Returns top 3 features with values without per-flow latency.
  4. Absence of manual form routes (manual form completely removed).
"""

import json
from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool

from api.database import Base, get_db
from api.main import app, startup_event
from features.extract import FEATURE_COLS

# Test DB in-memory with StaticPool to share connection across sessions
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
startup_event()
client = TestClient(app)


def test_no_manual_form_route():
    """Verify manual form route is completely removed."""
    response = client.get("/predict")
    assert response.status_code == 404
    response = client.post("/predict")
    assert response.status_code == 404


def test_schema_rejection_hard_fail_http_422():
    """Hard Constraint #3: Hard-fail with HTTP 422 on column mismatch/missing column."""
    # CSV missing required columns
    bad_csv = "src_ip,dst_ip,flow_duration\n192.168.1.1,10.0.0.1,100\n"
    response = client.post(
        "/api/score/csv",
        files={"file": ("bad_schema.csv", bad_csv, "text/csv")},
    )
    assert response.status_code == 422, "API must hard-fail with 422 on schema violation"
    assert "Schema validation hard-fail" in response.json()["detail"]


def test_valid_csv_scoring_and_aggregation():
    """Verify valid flow scoring and that repeated hits from same source aggregate."""
    # Generate 5 flows with identical 5-tuple (simulating a scan burst)
    rows = []
    for _ in range(5):
        row = {col: 100.0 for col in FEATURE_COLS}
        row["src_ip"] = "192.168.1.100"
        row["dst_ip"] = "10.0.0.5"
        row["dst_port"] = 8080
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_bytes = df.to_csv(index=False)

    response = client.post(
        "/api/score/csv",
        files={"file": ("valid_burst.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["flows_processed"] == 5

    # Check alert was aggregated
    alerts_response = client.get("/api/alerts")
    assert alerts_response.status_code == 200
    alerts = alerts_response.json()
    assert len(alerts) >= 1

    # Verify hit_count aggregated the burst
    matching = [a for a in alerts if a["src_ip"] == "192.168.1.100"]
    assert len(matching) == 1, "Burst must aggregate to a single alert row"
    assert matching[0]["hit_count"] == 5, f"Hit count should be 5, got {matching[0]['hit_count']}"


def test_lazy_shap_explanation():
    """Verify lazy SHAP explanation returns top 3 features with values."""
    alerts = client.get("/api/alerts").json()
    assert len(alerts) > 0
    alert_id = alerts[0]["id"]

    response = client.get(f"/api/alerts/{alert_id}/shap")
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert len(data["top_features"]) == 3
    for feat in data["top_features"]:
        assert "feature" in feat
        assert "value" in feat
        assert "shap_value" in feat
        assert feat["direction"] in ("push_attack", "push_benign")
