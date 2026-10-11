"""Phase 3 Verification Tests.

Validates the PyTorch Autoencoder, Isolation Forest baseline, zero-day experiment
metrics, and hybrid score fusion logic according to Hard Constraints #1 and #2.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest

from features.extract import FEATURE_COLS
from models.autoencoder import AutoencoderAnomalyDetector
from models.isolation_forest import IsolationForestAnomalyDetector
from models.hybrid import fuse_decision

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def test_autoencoder_artifacts_and_inference():
    """Verify saved autoencoder loads and computes reconstruction error."""
    assert (MODELS_DIR / "autoencoder_weights.pt").exists()
    assert (MODELS_DIR / "autoencoder_meta.json").exists()

    detector = AutoencoderAnomalyDetector.load(MODELS_DIR)
    assert detector.threshold > 0.0

    # Test inference on dummy flow
    dummy_data = {col: [1.0] for col in FEATURE_COLS}
    dummy_df = pd.DataFrame(dummy_data)

    raw_scores = detector.score(dummy_df)
    norm_scores = detector.score_normalized(dummy_df)
    preds = detector.predict(dummy_df)

    assert len(raw_scores) == 1
    assert 0.0 <= norm_scores[0] <= 1.0
    assert preds[0] in (0, 1)


def test_isolation_forest_baseline():
    """Verify isolation forest loads and scores properly."""
    assert (MODELS_DIR / "isolation_forest.pkl").exists()
    assert (MODELS_DIR / "if_threshold.json").exists()

    if_detector = IsolationForestAnomalyDetector.load(MODELS_DIR)
    dummy_data = {col: [1.0] for col in FEATURE_COLS}
    dummy_df = pd.DataFrame(dummy_data)

    scores = if_detector.score(dummy_df)
    preds = if_detector.predict(dummy_df)

    assert len(scores) == 1
    assert preds[0] in (0, 1)


def test_hybrid_score_fusion_matrix():
    """Verify all 4 cells of the hybrid decision matrix."""
    tau_sup = 0.8
    tau_anom = 0.5

    # 1. Critical: Supervised Attack + Anomaly High
    sev1, reason1 = fuse_decision(0.95, 0.85, tau_sup, tau_anom, "DoS")
    assert sev1 == "Critical"
    assert "High-confidence attack" in reason1

    # 2. High: Supervised Attack + Anomaly Low
    sev2, reason2 = fuse_decision(0.95, 0.20, tau_sup, tau_anom, "BruteForce")
    assert sev2 == "High"
    assert "Known attack signature" in reason2

    # 3. Medium: Supervised Benign + Anomaly High (Potential Zero-Day)
    sev3, reason3 = fuse_decision(0.10, 0.75, tau_sup, tau_anom, "BENIGN")
    assert sev3 == "Medium"
    assert "Zero-Day" in reason3

    # 4. Low: Supervised Benign + Anomaly Low (Normal)
    sev4, reason4 = fuse_decision(0.05, 0.15, tau_sup, tau_anom, "BENIGN")
    assert sev4 == "Low"
    assert "Benign network flow" in reason4


def test_zero_day_report_artifacts():
    """Verify Phase 3 experiment generated comprehensive metrics and report."""
    assert (REPORTS_DIR / "zero_day.md").exists()
    assert (REPORTS_DIR / "zero_day_metrics.json").exists()

    with open(REPORTS_DIR / "zero_day_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["held_out_class"] == "PortScan"
    assert data["autoencoder_anomaly"]["held_out_detection_rate"] >= 0.90
    assert data["autoencoder_anomaly"]["benign_false_alarm_rate"] <= 0.05
