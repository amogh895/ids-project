"""Phase 2 Verification Tests.

Validates the trained supervised pipeline, calibrated threshold selection,
and honest evaluation metrics.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from features.extract import FEATURE_COLS

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def test_pipeline_and_threshold_exist():
    """Verify saved model pipeline and threshold artifact exist."""
    pipeline_path = MODELS_DIR / "supervised_pipeline.pkl"
    threshold_path = MODELS_DIR / "calibrated_threshold.json"

    assert pipeline_path.exists(), "Supervised pipeline artifact missing"
    assert threshold_path.exists(), "Calibrated threshold artifact missing"

    with open(threshold_path, "r") as f:
        thresh_data = json.load(f)

    assert "threshold" in thresh_data
    assert thresh_data["fpr_target"] == 0.001
    assert 0.0 < thresh_data["threshold"] < 1.0


def test_pipeline_inference():
    """Verify fitted pipeline predicts on valid feature dataframe."""
    pipeline = joblib.load(MODELS_DIR / "supervised_pipeline.pkl")

    # Synthetic single sample with canonical features
    sample_data = {col: [0.0] for col in FEATURE_COLS}
    sample_df = pd.DataFrame(sample_data)

    proba = pipeline.predict_proba(sample_df)
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(), 1.0)


def test_evaluation_metrics_integrity():
    """Verify headline metric Recall @ 0.1% FPR and PR-AUC are logged properly."""
    metrics_path = REPORTS_DIR / "phase2_metrics.json"
    assert metrics_path.exists(), "Phase 2 metrics artifact missing"

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    assert "recall_at_fpr_0_1_pct" in metrics
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert metrics["recall_at_fpr_0_1_pct"] >= 0.90
    assert metrics["pr_auc"] >= 0.90
