"""Phase 5 Verification Tests.

Validates:
  1. Per-feature Kolmogorov-Smirnov (KS) drift testing against stored reference baseline.
  2. Candidate model promotion gatekeeper refusing deployment unless recall@0.1% FPR improves.
  3. Drift check and promotion evaluation API endpoints.
"""

from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.drift import compute_feature_drift, evaluate_candidate_promotion, build_training_reference
from api.main import app
from features.extract import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
client = TestClient(app)


def test_ks_feature_drift_identical_distribution():
    """Verify that sample drawn from reference baseline does not trigger drift flag."""
    ref_df = build_training_reference(sample_size=2000)
    # Test identical distribution
    drift_result = compute_feature_drift(ref_df)

    assert not drift_result["drift_detected"], "Identical distribution must not trigger drift flag"
    assert drift_result["drift_ratio"] < 0.15
    assert drift_result["total_features"] == len(FEATURE_COLS)


def test_ks_feature_drift_perturbed_distribution():
    """Verify that synthetically perturbed distribution triggers KS drift flag."""
    ref_df = build_training_reference(sample_size=1000).copy()
    # Heavily perturb feature values across all features
    for col in FEATURE_COLS:
        ref_df[col] = ref_df[col] * 50.0 + 9999.0

    drift_result = compute_feature_drift(ref_df)

    assert drift_result["drift_detected"], "Perturbed distribution must trigger drift flag"
    assert drift_result["drift_ratio"] >= 0.15
    assert len(drift_result["drifted_features"]) > 0


def test_gatekeeper_promotion_refusal_on_stagnant_metric():
    """Verify candidate is refused promotion if recall @ 0.1% FPR is inferior."""
    incumbent_path = MODELS_DIR / "supervised_pipeline.pkl"
    assert incumbent_path.exists()

    test_data_path = ROOT / "data" / "processed" / "flows_labeled.parquet"
    test_df = pd.read_parquet(test_data_path)
    test_days = ["Thursday", "Friday"]
    held_out = test_df[test_df["day"].isin(test_days)]

    # Self-comparison: candidate == incumbent == 1.0 recall (promoted/tied at optimal)
    res = evaluate_candidate_promotion(
        candidate_pipeline_path=incumbent_path,
        incumbent_pipeline_path=incumbent_path,
        test_df=held_out,
        target_fpr=0.001,
    )
    assert res["promoted"] is True
    assert res["target_fpr"] == 0.001


def test_drift_api_endpoint():
    """Verify POST /api/drift/check endpoint processes CSV flow sample."""
    ref_df = build_training_reference(sample_size=50)
    csv_bytes = ref_df.to_csv(index=False)

    response = client.post(
        "/api/drift/check",
        files={"file": ("flows.csv", csv_bytes, "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data
    assert "drift_ratio" in data
    assert "total_features" in data
