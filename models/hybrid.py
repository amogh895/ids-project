"""Hybrid Decision and Alert Severity Fusion Layer.

Combines the supervised classifier confidence score with the unsupervised
autoencoder anomaly score to generate a calibrated security alert with
actionable severity levels.
"""

from __future__ import annotations

from typing import Literal
import numpy as np

SeverityLevel = Literal["Critical", "High", "Medium", "Low"]


def fuse_decision(
    p_supervised: float,
    s_anomaly: float,
    threshold_supervised: float,
    threshold_anomaly: float = 0.5,
    predicted_class: str = "BENIGN",
) -> tuple[SeverityLevel, str]:
    """Fuse supervised probability and anomaly score into an alert severity.

    Decision Matrix:
    ┌───────────────────────┬──────────────────────────┬──────────────────────┐
    │                       │ Anomaly High (s >= tau)  │ Anomaly Low (s < tau)│
    ├───────────────────────┼──────────────────────────┼──────────────────────┤
    │ Supervised Attack     │ CRITICAL                 │ HIGH                 │
    │ (p >= threshold)      │ (Confirmed Threat)       │ (Known Signature)    │
    ├───────────────────────┼──────────────────────────┼──────────────────────┤
    │ Supervised Benign     │ MEDIUM                   │ LOW                  │
    │ (p < threshold)       │ (Potential Zero-Day)     │ (Normal Baseline)    │
    └───────────────────────┴──────────────────────────┴──────────────────────┘

    Returns:
        (severity, alert_reason)
    """
    is_sup_attack = p_supervised >= threshold_supervised
    is_anom_anomaly = s_anomaly >= threshold_anomaly

    if is_sup_attack and is_anom_anomaly:
        return (
            "Critical",
            f"High-confidence attack ({predicted_class}) with severe flow anomaly ({s_anomaly:.2f}).",
        )
    elif is_sup_attack and not is_anom_anomaly:
        return (
            "High",
            f"Known attack signature detected ({predicted_class}) with standard packet dynamics.",
        )
    elif not is_sup_attack and is_anom_anomaly:
        return (
            "Medium",
            f"Potential Zero-Day / Novel anomaly: bypassed supervised detection but exhibits abnormal flow reconstruction error ({s_anomaly:.2f}).",
        )
    else:
        return (
            "Low",
            "Benign network flow within normal statistical boundaries.",
        )
