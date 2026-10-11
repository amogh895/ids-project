"""Alert Aggregator — Problem #4: Alert Fatigue.

Groups high-volume alerts by:
  (source_ip, destination_port_pattern, time_window)

So a port scan hitting 10,000 ports yields ONE aggregated alert with hit_count=10000,
rather than 10,000 separate DB rows, preventing SOC dashboard collapse.
"""

from __future__ import annotations

import datetime
from typing import Optional
from sqlalchemy.orm import Session
from api.database import Alert

WINDOW_SECONDS = 60  # Aggregation window in seconds


def _port_pattern(dst_port: Optional[int]) -> str:
    """Bucket destination ports into common attack patterns.
    
    Well-known services get exact buckets, high ephemeral/scanning ports get grouped.
    """
    if dst_port is None:
        return "unknown"
    if dst_port in (80, 443, 8080):
        return "web"
    if dst_port in (22, 2222):
        return "ssh"
    if dst_port in (21, 20):
        return "ftp"
    if dst_port in (25, 465, 587):
        return "smtp"
    if dst_port in (53,):
        return "dns"
    if dst_port in (3389,):
        return "rdp"
    if dst_port in (445, 139):
        return "smb"
    # Port scan pattern: high or varying ephemeral ports
    return "ephemeral_scan"


def make_agg_key(src_ip: str, dst_port: Optional[int], attack_class: str) -> str:
    """Generate deterministic aggregation key."""
    pattern = _port_pattern(dst_port)
    return f"{src_ip}::{pattern}::{attack_class}"


def aggregate_and_record_alert(
    db: Session,
    flow_meta: dict,
    predicted_class: str,
    supervised_score: float,
    anomaly_score: float,
    severity: str,
    alert_reason: str,
    model_version: str = "1.0.0",
) -> Alert:
    """Find existing open alert in time window or create a new one.
    
    If an alert with the same (src_ip, port_pattern, attack_class) was created
    within WINDOW_SECONDS, increment hit_count and update timestamp instead of
    inserting a duplicate row.
    """
    src_ip = str(flow_meta.get("src_ip", "0.0.0.0"))
    dst_ip = str(flow_meta.get("dst_ip", "0.0.0.0"))
    src_port = flow_meta.get("src_port")
    dst_port = flow_meta.get("dst_port")
    protocol = flow_meta.get("protocol")

    agg_key = make_agg_key(src_ip, dst_port, predicted_class)
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=WINDOW_SECONDS)

    existing: Optional[Alert] = (
        db.query(Alert)
        .filter(
            Alert.agg_key == agg_key,
            Alert.updated_at >= cutoff,
        )
        .order_by(Alert.updated_at.desc())
        .first()
    )

    if existing is not None:
        existing.hit_count += 1
        existing.updated_at = datetime.datetime.utcnow()
        # Keep maximum observed scores
        if supervised_score > existing.supervised_score:
            existing.supervised_score = supervised_score
        if anomaly_score > existing.anomaly_score:
            existing.anomaly_score = anomaly_score
        db.commit()
        db.refresh(existing)
        return existing

    # Create new alert
    alert = Alert(
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        predicted_class=predicted_class,
        supervised_score=supervised_score,
        anomaly_score=anomaly_score,
        severity=severity,
        alert_reason=alert_reason,
        hit_count=1,
        agg_key=agg_key,
        model_version=model_version,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert
