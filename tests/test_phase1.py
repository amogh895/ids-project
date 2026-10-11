"""Phase 1 Verification Tests.

Validates schema contract adherence, flow extraction integrity, and leak-free
dataset generation according to Hard Constraints #1, #2, and #3.
"""

from pathlib import Path
import pytest
import pandas as pd
from scapy.layers.inet import IP, TCP
from scapy.packet import Packet
from scapy.utils import wrpcap

from features.extract import (
    FEATURE_COLS,
    METADATA_COLS,
    extract_flows,
    load_schema,
    validate_inference_schema,
)
from data.label import generate_representative_dataset, generate_data_quality_report


def test_schema_contract_structure():
    """Verify frozen schema specification exists and conforms to standard."""
    schema = load_schema()
    assert schema["schema_version"] == "1.0.0"
    assert len(schema["feature_columns"]) == 22
    assert len(schema["metadata_columns"]) == 8

    # Verify all feature columns have valid dtypes
    for col in schema["feature_columns"]:
        assert col["dtype"] in ("float64", "int64")


def test_schema_validation_hard_fail():
    """Verify inference schema validation hard-fails on missing or reordered columns (Constraint #3)."""
    # 1. Missing column
    df_missing = pd.DataFrame(columns=FEATURE_COLS[:-1])
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_inference_schema(df_missing)

    # 2. Extra unexpected column
    df_extra = pd.DataFrame(columns=FEATURE_COLS + ["unexpected_col"])
    with pytest.raises(ValueError, match="Schema validation failed"):
        validate_inference_schema(df_extra)

    # 3. Reordered columns
    reordered_cols = list(reversed(FEATURE_COLS))
    df_reordered = pd.DataFrame(columns=reordered_cols)
    with pytest.raises(ValueError, match="Columns not in canonical order"):
        validate_inference_schema(df_reordered)

    # 4. Valid canonical schema passes
    df_valid = pd.DataFrame(columns=FEATURE_COLS)
    validate_inference_schema(df_valid)


def test_flow_extraction_from_pcap(tmp_path: Path):
    """Verify extract_flows parses pcap and emits exact canonical schema (Constraint #2)."""
    pcap_file = tmp_path / "test_sample.pcap"

    # Synthesize two packets in a bidirectional TCP flow
    pkt1 = IP(src="192.168.1.10", dst="192.168.1.50") / TCP(sport=12345, dport=80, flags="S", window=29200)
    pkt2 = IP(src="192.168.1.50", dst="192.168.1.10") / TCP(sport=80, dport=12345, flags="SA", window=29200)
    wrpcap(str(pcap_file), [pkt1, pkt2])

    df_flows = extract_flows(pcap_file)
    assert len(df_flows) == 1
    assert df_flows.iloc[0]["src_ip"] == "192.168.1.10"
    assert df_flows.iloc[0]["dst_port"] == 80
    assert df_flows.iloc[0]["fwd_packets"] == 1
    assert df_flows.iloc[0]["bwd_packets"] == 1

    # Extract features only and validate canonical schema
    df_features = df_flows[FEATURE_COLS]
    validate_inference_schema(df_features)


def test_representative_dataset_integrity():
    """Verify dataset generator adheres to temporal distribution and quality standards."""
    df = generate_representative_dataset(n_samples=1000, seed=42)
    assert len(df) == 1000

    # Ensure timestamp monotonicity (time-ordered)
    assert df["timestamp"].is_monotonic_increasing

    # Ensure no NaN or infinite values
    assert df[FEATURE_COLS].isna().sum().sum() == 0

    # Check data quality report
    report = generate_data_quality_report(df)
    assert report["missing_values"] == 0
    assert report["infinite_values"] == 0
    assert report["total_flows"] == 1000
    assert report["attack_prevalence_pct"] > 0.0
