"""Lazy SHAP Explanation Module.

Only computes SHAP TreeExplainer values on-demand when an analyst clicks
an alert in the dashboard — NEVER per flow during scoring — saving CPU/latency.
Returns the top 3 contributing features with values and impact directions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
import shap

from features.extract import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

_tree_explainer: Optional[shap.TreeExplainer] = None
_pipeline: Optional[Any] = None


def _get_explainer():
    """Lazily initialize TreeExplainer on first request."""
    global _tree_explainer, _pipeline
    if _tree_explainer is None:
        pipeline_path = MODELS_DIR / "supervised_pipeline.pkl"
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Trained pipeline missing at {pipeline_path}")
        _pipeline = joblib.load(pipeline_path)
        # Extract underlying LightGBM classifier from sklearn pipeline
        classifier = _pipeline.named_steps["classifier"]
        _tree_explainer = shap.TreeExplainer(classifier)
    return _tree_explainer, _pipeline


def explain_flow(flow_dict: dict, top_k: int = 3) -> list[dict]:
    """Compute top-k SHAP feature contributions for a single flow dict.
    
    Args:
        flow_dict: dictionary containing the 22 FEATURE_COLS
        top_k: number of top contributing features to return (default: 3)
        
    Returns:
        list of dicts: [
          {"feature": "fwd_pkts_per_sec", "value": 1420.5, "shap_value": 0.42, "direction": "push_attack"},
          ...
        ]
    """
    explainer, pipeline = _get_explainer()

    # Preprocess flow using the fitted pipeline's preprocessor
    df_raw = pd.DataFrame([{col: flow_dict.get(col, 0.0) for col in FEATURE_COLS}])
    preprocessor = pipeline.named_steps["preprocessor"]
    x_transformed = preprocessor.transform(df_raw)

    shap_values = explainer.shap_values(x_transformed)
    # Binary classification in TreeExplainer: shap_values can be array or list
    if isinstance(shap_values, list):
        # shap_values[1] is the log-odds impact towards class 1 (ATTACK)
        sv = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    # Rank features by absolute impact
    indices = np.argsort(np.abs(sv))[::-1][:top_k]

    results = []
    for idx in indices:
        feature_name = FEATURE_COLS[idx]
        val = float(flow_dict.get(feature_name, 0.0))
        impact = float(sv[idx])
        results.append({
            "feature": feature_name,
            "value": round(val, 4),
            "shap_value": round(impact, 4),
            "direction": "push_attack" if impact > 0 else "push_benign",
        })

    return results
