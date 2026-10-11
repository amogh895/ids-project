"""Flow Extraction Engine.

Exposes a unified function `extract_flows` that converts raw packet captures (PCAP)
or live interface traffic into bidirectional network flows with a frozen,
strictly-enforced schema contract (features/schema_v1.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


SCHEMA_PATH = Path(__file__).resolve().parent / "schema_v1.json"


def load_schema() -> dict[str, Any]:
    """Load the frozen v1 feature schema contract."""
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


SCHEMA = load_schema()
METADATA_COLS: list[str] = [c["name"] for c in SCHEMA["metadata_columns"]]
FEATURE_COLS: list[str] = [c["name"] for c in SCHEMA["feature_columns"]]
FEATURE_DTYPES: dict[str, str] = {c["name"]: c["dtype"] for c in SCHEMA["feature_columns"]}


def get_feature_names() -> list[str]:
    """Return the ordered list of canonical feature column names."""
    return list(FEATURE_COLS)


def validate_inference_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Hard-fail (ValueError) on any column mismatch, column order, or missing columns.

    Used at inference to guarantee schema contract enforcement (Constraint #3).
    Allows recognized metadata columns (e.g. 5-tuple) alongside canonical features,
    but strictly rejects any unknown columns, missing features, or feature order mismatch.
    """
    incoming_cols = list(df.columns)
    incoming_features = [c for c in incoming_cols if c in FEATURE_COLS]
    missing_features = [c for c in FEATURE_COLS if c not in incoming_cols]
    unknown_cols = [c for c in incoming_cols if c not in FEATURE_COLS and c not in METADATA_COLS and c not in ("label", "timetable_window")]

    if missing_features or unknown_cols or incoming_features != FEATURE_COLS:
        msg = f"Schema validation failed (Constraint #3). Missing features: {missing_features}, Unknown columns: {unknown_cols}"
        if incoming_features != FEATURE_COLS and not missing_features:
            msg += f". Columns not in canonical order. Expected: {FEATURE_COLS[:3]}..., Got: {incoming_features[:3]}..."
        raise ValueError(msg)
    return df


def _scapy_extract_flows(pcap_path: str | Path) -> pd.DataFrame:
    """Extract flows from PCAP using Scapy pure-Python parser.

    Guarantees deterministic bidirectional flow calculation and feature parity
    without requiring OS-level packet drivers.
    """
    from scapy.utils import PcapReader
    from scapy.layers.inet import IP, TCP, UDP

    pcap_path = str(pcap_path)
    flows: dict[tuple, dict[str, Any]] = {}

    with PcapReader(pcap_path) as pcap_reader:
        for pkt in pcap_reader:
            if not pkt.haslayer(IP):
                continue

            ip = pkt[IP]
            proto = int(ip.proto)
            src_ip = ip.src
            dst_ip = ip.dst
            pkt_len = len(pkt)
            ts = float(pkt.time)

            src_port = 0
            dst_port = 0
            flags = ""
            win_size = 0

            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                src_port = int(tcp.sport)
                dst_port = int(tcp.dport)
                flags = str(tcp.flags)
                win_size = int(tcp.window)
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                src_port = int(udp.sport)
                dst_port = int(udp.dport)

            # Bidirectional canonical key: forward is lower IP/port tuple
            fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
            rev_key = (dst_ip, src_ip, dst_port, src_port, proto)

            if fwd_key in flows:
                flow = flows[fwd_key]
                direction = "fwd"
            elif rev_key in flows:
                flow = flows[rev_key]
                direction = "bwd"
            else:
                flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{proto}"
                flows[fwd_key] = {
                    "flow_id": flow_id,
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "protocol": proto,
                    "start_time": ts,
                    "last_time": ts,
                    "fwd_packets": 0,
                    "bwd_packets": 0,
                    "fwd_bytes": 0,
                    "bwd_bytes": 0,
                    "fwd_lengths": [],
                    "bwd_lengths": [],
                    "all_iats": [],
                    "fwd_iats": [],
                    "bwd_iats": [],
                    "last_fwd_time": None,
                    "last_bwd_time": None,
                    "syn_flag_count": 0,
                    "fin_flag_count": 0,
                    "rst_flag_count": 0,
                    "psh_flag_count": 0,
                    "ack_flag_count": 0,
                    "init_win_bytes_fwd": win_size,
                    "init_win_bytes_bwd": 0,
                }
                flow = flows[fwd_key]
                direction = "fwd"

            # Update timing & metrics
            time_since_last = ts - flow["last_time"]
            if time_since_last >= 0:
                flow["all_iats"].append(time_since_last)
            flow["last_time"] = max(flow["last_time"], ts)

            if direction == "fwd":
                flow["fwd_packets"] += 1
                flow["fwd_bytes"] += pkt_len
                flow["fwd_lengths"].append(pkt_len)
                if flow["last_fwd_time"] is not None:
                    flow["fwd_iats"].append(ts - flow["last_fwd_time"])
                flow["last_fwd_time"] = ts
            else:
                flow["bwd_packets"] += 1
                flow["bwd_bytes"] += pkt_len
                flow["bwd_lengths"].append(pkt_len)
                if flow["init_win_bytes_bwd"] == 0 and win_size > 0:
                    flow["init_win_bytes_bwd"] = win_size
                if flow["last_bwd_time"] is not None:
                    flow["bwd_iats"].append(ts - flow["last_bwd_time"])
                flow["last_bwd_time"] = ts

            # TCP Flags
            if "S" in flags:
                flow["syn_flag_count"] += 1
            if "F" in flags:
                flow["fin_flag_count"] += 1
            if "R" in flags:
                flow["rst_flag_count"] += 1
            if "P" in flags:
                flow["psh_flag_count"] += 1
            if "A" in flags:
                flow["ack_flag_count"] += 1

    records = []
    for f in flows.values():
        duration = max(0.000001, f["last_time"] - f["start_time"])
        tot_bytes = f["fwd_bytes"] + f["bwd_bytes"]
        tot_pkts = f["fwd_packets"] + f["bwd_packets"]

        fwd_len_mean = float(np.mean(f["fwd_lengths"])) if f["fwd_lengths"] else 0.0
        bwd_len_mean = float(np.mean(f["bwd_lengths"])) if f["bwd_lengths"] else 0.0
        fwd_len_std = float(np.std(f["fwd_lengths"])) if len(f["fwd_lengths"]) > 1 else 0.0
        bwd_len_std = float(np.std(f["bwd_lengths"])) if len(f["bwd_lengths"]) > 1 else 0.0

        flow_iat_mean = float(np.mean(f["all_iats"])) if f["all_iats"] else 0.0
        flow_iat_std = float(np.std(f["all_iats"])) if len(f["all_iats"]) > 1 else 0.0
        fwd_iat_mean = float(np.mean(f["fwd_iats"])) if f["fwd_iats"] else 0.0
        bwd_iat_mean = float(np.mean(f["bwd_iats"])) if f["bwd_iats"] else 0.0

        records.append({
            # Metadata
            "flow_id": f["flow_id"],
            "src_ip": f["src_ip"],
            "dst_ip": f["dst_ip"],
            "src_port": int(f["src_port"]),
            "dst_port": int(f["dst_port"]),
            "protocol": int(f["protocol"]),
            "timestamp": float(f["start_time"]),
            "day": "Unknown",
            # Features strictly ordered
            "flow_duration": float(duration),
            "fwd_packets": int(f["fwd_packets"]),
            "bwd_packets": int(f["bwd_packets"]),
            "fwd_bytes": int(f["fwd_bytes"]),
            "bwd_bytes": int(f["bwd_bytes"]),
            "fwd_pkt_len_mean": float(fwd_len_mean),
            "bwd_pkt_len_mean": float(bwd_len_mean),
            "fwd_pkt_len_std": float(fwd_len_std),
            "bwd_pkt_len_std": float(bwd_len_std),
            "flow_bytes_per_sec": float(tot_bytes / duration),
            "flow_pkts_per_sec": float(tot_pkts / duration),
            "flow_iat_mean": float(flow_iat_mean),
            "flow_iat_std": float(flow_iat_std),
            "fwd_iat_mean": float(fwd_iat_mean),
            "bwd_iat_mean": float(bwd_iat_mean),
            "syn_flag_count": int(f["syn_flag_count"]),
            "fin_flag_count": int(f["fin_flag_count"]),
            "rst_flag_count": int(f["rst_flag_count"]),
            "psh_flag_count": int(f["psh_flag_count"]),
            "ack_flag_count": int(f["ack_flag_count"]),
            "init_win_bytes_fwd": int(f["init_win_bytes_fwd"]),
            "init_win_bytes_bwd": int(f["init_win_bytes_bwd"]),
        })

    df = pd.DataFrame(records)
    return df


def extract_flows(source: str | Path, is_interface: bool = False) -> pd.DataFrame:
    """Primary entry point for flow extraction.

    Takes a pcap file path or live interface name and returns a DataFrame
    conforming strictly to the frozen schema contract.
    """
    if is_interface:
        # Live interface capture stub
        raise NotImplementedError("Live interface capture is supported on Linux with libpcap/nfstream.")

    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"PCAP source not found: {source_path}")

    # Use Scapy flow parser
    df = _scapy_extract_flows(source_path)

    # Cast types strictly to match schema
    for col in METADATA_COLS:
        if col in df.columns:
            if col in ("src_port", "dst_port", "protocol"):
                df[col] = df[col].astype("int64")
            elif col == "timestamp":
                df[col] = df[col].astype("float64")
            else:
                df[col] = df[col].astype("string")

    for col in FEATURE_COLS:
        dtype = FEATURE_DTYPES[col]
        df[col] = df[col].astype(dtype)

    return df
