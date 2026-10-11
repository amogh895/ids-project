"""Drift Monitoring Engine — Problem #3: Concept Drift.

Implements:
  1. Per-feature two-sample Kolmogorov-Smirnov (KS) test comparing incoming
     production flows against a stored reference training distribution.
  2. Prediction confidence distribution monitoring.
  3. Candidate model retraining evaluation that refuses promotion unless
     Recall @ 0.1% FPR strictly improves over the incumbent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from features.extract import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DATA_PATH = ROOT / "data" / "processed" / "flows_labeled.parquet"

REFERENCE_PATH = REPORTS_DIR / "training_reference_stats.parquet"


def build_training_reference(sample_size: int = 5000) -> pd.DataFrame:
    """Extract and persist reference distribution from benign training flows."""
    if REFERENCE_PATH.exists():
        return pd.read_parquet(REFERENCE_PATH)

    df = pd.read_parquet(DATA_PATH)
    benign_df = df[df["label"] == "BENIGN"][FEATURE_COLS]
    ref_df = benign_df.sample(n=min(sample_size, len(benign_df)), random_state=42)
    ref_df.to_parquet(REFERENCE_PATH, index=False)
    return ref_df


def compute_feature_drift(incoming_df: pd.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Run two-sample Kolmogorov-Smirnov test per feature against stored reference.

    Returns:
        {
          "drift_detected": bool,
          "drifted_features": ["fwd_pkts_per_sec", ...],
          "drift_ratio": 0.18,
          "feature_p_values": {"feature_name": p_value, ...}
        }
    """
    ref_df = build_training_reference()
    drifted_features: list[str] = []
    p_values: dict[str, float] = {}

    for col in FEATURE_COLS:
        if col not in incoming_df.columns:
            continue
        ref_vals = ref_df[col].dropna().values
        cur_vals = incoming_df[col].dropna().values

        if len(cur_vals) < 5 or len(ref_vals) < 5:
            continue

        stat, p_val = ks_2samp(ref_vals, cur_vals)
        p_values[col] = float(p_val)
        if p_val < alpha:
            drifted_features.append(col)

    drift_ratio = len(drifted_features) / max(1, len(FEATURE_COLS))
    # Drift flag triggers if > 15% of features exhibit statistically significant drift
    drift_detected = drift_ratio >= 0.15

    return {
        "drift_detected": drift_detected,
        "drift_ratio": round(drift_ratio, 4),
        "drifted_feature_count": len(drifted_features),
        "total_features": len(FEATURE_COLS),
        "drifted_features": drifted_features,
        "feature_p_values": {k: round(v, 6) for k, v in p_values.items()},
    }


def evaluate_candidate_promotion(
    candidate_pipeline_path: Path,
    incumbent_pipeline_path: Path,
    test_df: pd.DataFrame,
    target_fpr: float = 0.001,
) -> dict[str, Any]:
    """Automated gatekeeper: Refuses model promotion unless Recall @ 0.1% FPR improves.

    Evaluated strictly on the frozen held-out test split.
    """
    from sklearn.metrics import roc_curve

    cand_pipeline = joblib.load(candidate_pipeline_path)
    inc_pipeline = joblib.load(incumbent_pipeline_path)

    y_test = (test_df["label"] != "BENIGN").astype(int).values
    X_test = test_df[FEATURE_COLS]

    cand_proba = cand_pipeline.predict_proba(X_test)[:, 1]
    inc_proba = inc_pipeline.predict_proba(X_test)[:, 1]

    # Incumbent Recall @ target_fpr
    fpr_inc, tpr_inc, _ = roc_curve(y_test, inc_proba)
    idx_inc = np.searchsorted(fpr_inc, target_fpr, side="right") - 1
    idx_inc = max(0, min(idx_inc, len(tpr_inc) - 1))
    incumbent_recall = float(tpr_inc[idx_inc])

    # Candidate Recall @ target_fpr
    fpr_cand, tpr_cand, _ = roc_curve(y_test, cand_proba)
    idx_cand = np.searchsorted(fpr_cand, target_fpr, side="right") - 1
    idx_cand = max(0, min(idx_cand, len(tpr_cand) - 1))
    candidate_recall = float(tpr_cand[idx_cand])

    # Gatekeeper condition: must strictly improve or match 1.0
    promoted = candidate_recall > incumbent_recall or (
        candidate_recall == 1.0 and incumbent_recall == 1.0
    )

    return {
        "promoted": promoted,
        "target_fpr": target_fpr,
        "incumbent_recall_at_fpr": round(incumbent_recall, 4),
        "candidate_recall_at_fpr": round(candidate_recall, 4),
        "decision": (
            "PROMOTED: Candidate achieved superior/optimal recall @ 0.1% FPR."
            if promoted
            else "REJECTED: Candidate failed to improve recall @ 0.1% FPR over incumbent."
        ),
    }
