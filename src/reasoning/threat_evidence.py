"""
M4 - Threat Evidence Builder
============================
Builds the JSON threat-evidence object (M4 -> M5 contract) from REAL SAFE
detections.

Inputs
  - results/safe/<dataset>/safe_predictions.csv   (SAFE scores, verdicts,
    test-row index and top-feature explanations; written by safe.py)
  - results/safe/<dataset>/safe_metadata.json     (threshold)
  - data/raw/<dataset>/...csv                     (original flow records:
    IPs, ports, protocol, packet counts, flags)
  - data/processed/<dataset>/metadata.json        (split seed / test size)

Output
  - results/reasoning/<dataset>/threat_evidence.json

What is real and what is not
  - anomaly score, threshold, verdict and explanation come from SAFE
  - 5-tuple and flow statistics come from the raw dataset record of that flow
    (fields the dataset does not have are reported as null, never invented)
  - device_context is an ASSUMPTION inferred from the destination port
  - ground_truth is stored for evaluation only; M5 never sees it

Linking a prediction back to its raw record: preprocess.py assigns each raw
row to train/test with a deterministic hash of (row index, label, seed). The
same hash is replayed here to map test.csv row k -> raw row index.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
SAFE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "safe")
REASONING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "reasoning")

# Must match preprocess.py
RAW_FILES = {
    "IoMT-TrafficData": os.path.join(PROJECT_ROOT, "data", "raw", "IoMT-TrafficData", "output.csv"),
    "MedSec-25": os.path.join(PROJECT_ROOT, "data", "raw", "MedSec-25", "MedSec-25.csv"),
}
RAW_LABEL_COLUMNS = {"IoMT-TrafficData": "is_attack", "MedSec-25": "Label"}
# Column that names the attack type in the raw file (IoMT: 'traffic')
RAW_ATTACK_TYPE_COLUMNS = {"IoMT-TrafficData": "traffic", "MedSec-25": "Label"}
PREPROCESS_CHUNK_SIZE = 50000

SAMPLE_SEED = 42

# Raw column names tried for each evidence field (first match wins)
FIELD_CANDIDATES = {
    "source_ip": ["id.orig_h", "Src IP", "src_ip", "Source IP", "srcip"],
    "destination_ip": ["id.resp_h", "Dst IP", "dst_ip", "Destination IP", "dstip"],
    "source_port": ["id.orig_p", "Src Port", "src_port", "Source Port", "sport"],
    "destination_port": ["id.resp_p", "Dst Port", "dst_port", "Destination Port", "dport"],
    "protocol": ["proto", "Protocol", "protocol"],
    "service": ["service", "Service"],
    "timestamp": ["ts", "Timestamp", "timestamp"],
    "flow_duration": ["flow_duration", "Flow Duration", "duration"],
    "fwd_packets": ["fwd_pkts_tot", "Tot Fwd Pkts", "Total Fwd Packets"],
    "bwd_packets": ["bwd_pkts_tot", "Tot Bwd Pkts", "Total Backward Packets"],
    "packets_per_sec": ["flow_pkts_per_sec", "Flow Pkts/s", "Flow Packets/s"],
    "bytes_per_sec": ["payload_bytes_per_second", "Flow Byts/s", "Flow Bytes/s"],
    "syn_flags": ["flow_SYN_flag_count", "SYN Flag Cnt", "SYN Flag Count"],
    "ack_flags": ["flow_ACK_flag_count", "ACK Flag Cnt", "ACK Flag Count"],
    "rst_flags": ["flow_RST_flag_count", "RST Flag Cnt", "RST Flag Count"],
    "fin_flags": ["flow_FIN_flag_count", "FIN Flag Cnt", "FIN Flag Count"],
}
PROTOCOL_NUMBERS = {1: "ICMP", 6: "TCP", 17: "UDP"}


# ============================================================
# CLINICAL IOMT DEVICE CONTEXT (assumed from destination port)
# ============================================================

IOMT_DEVICE_MAPPING = {
    80: {"role": "Infusion Pump Management Gateway / Clinical Web Portal", "device_type": "Smart Infusion Pump API",
         "criticality": "CRITICAL", "protocols": ["HTTP", "REST"]},
    443: {"role": "Electronic Health Record (EHR) Web Gateway", "device_type": "EHR / Clinical Telemetry Gateway",
          "criticality": "CRITICAL", "protocols": ["HTTPS", "TLS"]},
    1883: {"role": "Bedside Vitals Telemetry Message Broker", "device_type": "IoMT Sensor MQTT Gateway",
           "criticality": "HIGH", "protocols": ["MQTT"]},
    8883: {"role": "Secure Bedside Telemetry Message Broker", "device_type": "Secure IoMT Sensor MQTT Gateway",
           "criticality": "HIGH", "protocols": ["MQTTS", "TLS"]},
    104: {"role": "PACS Medical Imaging Archive / Server", "device_type": "DICOM Imaging Server",
          "criticality": "HIGH", "protocols": ["DICOM"]},
    11112: {"role": "PACS Medical Imaging Workstation", "device_type": "DICOM Modality Node",
            "criticality": "HIGH", "protocols": ["DICOM"]},
    2575: {"role": "HL7 Clinical Integration Engine", "device_type": "Hospital Information System (HIS) Gateway",
           "criticality": "CRITICAL", "protocols": ["HL7"]},
    22: {"role": "Clinical Appliance Management Shell", "device_type": "Medical Gateway Terminal",
         "criticality": "HIGH", "protocols": ["SSH"]},
    445: {"role": "Nurse Station Workstation / EHR Endpoint", "device_type": "Windows Clinical Terminal",
          "criticality": "HIGH", "protocols": ["SMB"]},
    3389: {"role": "Tele-ICU Remote Desktop Terminal", "device_type": "Remote Clinical Console",
           "criticality": "HIGH", "protocols": ["RDP"]},
}
DEFAULT_IOMT_CONTEXT = {"role": "Generic Clinical Medical Sensor / IoMT Node", "device_type": "IoMT End-Node",
                        "criticality": "MEDIUM", "protocols": []}


# ============================================================
# HELPERS
# ============================================================

def _py(value):
    """Convert pandas/numpy values to JSON-safe Python values (NaN -> None)."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _replay_split_hash(index: int, label: str, seed: int) -> float:
    """Identical to preprocess.random_value()."""
    text = f"{label}_{index}_{seed}"
    value = int(hashlib.md5(text.encode()).hexdigest(), 16)
    return (value % 1000000) / 1000000


def map_test_rows_to_raw(raw_path: str, label_column: str, seed: int, test_size: float) -> np.ndarray:
    """Raw row index of every test.csv row, in test.csv order."""
    raw_indices = []
    global_index = 0
    for chunk in pd.read_csv(raw_path, usecols=[label_column], chunksize=PREPROCESS_CHUNK_SIZE, low_memory=False):
        for label in chunk[label_column].fillna("UNKNOWN").astype(str).values:
            if _replay_split_hash(global_index, label, seed) < test_size:
                raw_indices.append(global_index)
            global_index += 1
    return np.asarray(raw_indices, dtype=np.int64)


def count_data_rows(csv_path: str) -> int:
    with open(csv_path, "rb") as f:
        return sum(1 for _ in f) - 1


# ============================================================
# THREAT EVIDENCE BUILDER
# ============================================================

class ThreatEvidenceBuilder:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_dir = os.path.join(DATA_DIR, dataset_name)
        self.metadata = self._load_json(os.path.join(self.dataset_dir, "metadata.json"))
        self.safe_metadata = self._load_json(os.path.join(SAFE_RESULTS_DIR, dataset_name, "safe_metadata.json"))
        self.predictions_path = os.path.join(SAFE_RESULTS_DIR, dataset_name, "safe_predictions.csv")

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    # --------------------------------------------------------
    # SAFE predictions
    # --------------------------------------------------------

    def _load_predictions(self) -> pd.DataFrame:
        if not os.path.exists(self.predictions_path):
            raise FileNotFoundError(
                f"SAFE predictions not found: {self.predictions_path}\nRun src/safe/safe.py first.")
        df = pd.read_csv(self.predictions_path)
        if "test_row_index" not in df.columns or (df["test_row_index"] < 0).all():
            raise RuntimeError(
                "safe_predictions.csv has no test_row_index column. Re-run src/safe/safe.py "
                "(the updated version) so predictions can be linked to flow records.")
        return df

    def _sample(self, df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
        """Half SAFE-flagged flows, half SAFE-normal flows (random, seeded)."""
        rng = np.random.default_rng(SAMPLE_SEED)
        flagged = df[df["predicted_binary_label"] == 1]
        normal = df[df["predicted_binary_label"] == 0]
        n_flag = min(len(flagged), max_samples // 2)
        n_norm = min(len(normal), max_samples - n_flag)
        n_flag = min(len(flagged), max_samples - n_norm)
        pick = []
        if n_flag:
            pick.append(flagged.iloc[rng.choice(len(flagged), n_flag, replace=False)])
        if n_norm:
            pick.append(normal.iloc[rng.choice(len(normal), n_norm, replace=False)])
        out = pd.concat(pick)
        return out.iloc[rng.permutation(len(out))].reset_index(drop=True)

    # --------------------------------------------------------
    # Raw records
    # --------------------------------------------------------

    def _load_raw_records(self, test_rows: np.ndarray) -> Dict[int, pd.Series]:
        """test.csv row -> raw record (empty dict if the mapping cannot be verified)."""
        raw_path = RAW_FILES.get(self.dataset_name)
        label_col = RAW_LABEL_COLUMNS.get(self.dataset_name)
        test_csv = os.path.join(self.dataset_dir, "test.csv")
        if not raw_path or not os.path.exists(raw_path):
            print(f"[M4] Raw file not found ({raw_path}); flow fields will be null.")
            return {}

        seed = int(self.metadata.get("random_seed", 42))
        test_size = float(self.metadata.get("test_size", 0.20))
        print("[M4] Linking test rows to raw flow records (replaying the preprocess split)...")
        mapping = map_test_rows_to_raw(raw_path, label_col, seed, test_size)
        n_test = count_data_rows(test_csv)
        if len(mapping) != n_test:
            print(f"[M4] WARNING: replayed split has {len(mapping):,} test rows but test.csv has "
                  f"{n_test:,}. Cannot link reliably; flow fields will be null.")
            return {}

        wanted = {int(mapping[r]): int(r) for r in test_rows}
        records = {}
        for chunk in pd.read_csv(raw_path, chunksize=PREPROCESS_CHUNK_SIZE, low_memory=False):
            hit = chunk.index.intersection(list(wanted.keys()))
            for raw_idx in hit:
                records[wanted[int(raw_idx)]] = chunk.loc[raw_idx]
            if len(records) == len(wanted):
                break
        print(f"[M4] Linked {len(records)}/{len(wanted)} flows to raw records.")
        return records

    @staticmethod
    def _field(raw: Optional[pd.Series], name: str):
        if raw is None:
            return None
        for col in FIELD_CANDIDATES[name]:
            if col in raw.index:
                return _py(raw[col])
        return None

    def _network_flow(self, raw: Optional[pd.Series]) -> Dict[str, Any]:
        flow = {k: self._field(raw, k) for k in FIELD_CANDIDATES if k != "timestamp"}
        for port in ("source_port", "destination_port"):
            try:
                flow[port] = int(float(flow[port])) if flow[port] is not None else None
            except (TypeError, ValueError):
                pass
        proto = flow.get("protocol")
        if isinstance(proto, (int, float)) and proto is not None:
            flow["protocol"] = PROTOCOL_NUMBERS.get(int(proto), str(int(proto)))
        elif isinstance(proto, str):
            flow["protocol"] = proto.upper()
        return flow

    def _ground_truth(self, pred_row: pd.Series, raw: Optional[pd.Series]) -> Dict[str, Any]:
        is_attack = bool(int(pred_row["true_binary_label"]) == 1)
        label = None
        type_col = RAW_ATTACK_TYPE_COLUMNS.get(self.dataset_name)
        if raw is not None and type_col in raw.index:
            label = str(raw[type_col])
        if label is None and self.dataset_name == "MedSec-25":
            inv = {v: k for k, v in self.metadata.get("label_mapping", {}).items()}
            label = inv.get(int(pred_row["true_original_label"]))
        if label is None:
            label = "attack" if is_attack else "normal"
        return {"is_attack": is_attack, "label": label}

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def load_flagged_flows(self, max_samples: int = 100) -> List[Dict[str, Any]]:
        preds = self._sample(self._load_predictions(), max_samples)
        raw_records = self._load_raw_records(preds["test_row_index"].to_numpy())
        threshold = float(self.safe_metadata.get("threshold", np.nan))

        items = []
        for i, p in preds.iterrows():
            test_row = int(p["test_row_index"])
            raw = raw_records.get(test_row)
            flow = self._network_flow(raw)

            port = flow.get("destination_port")
            device = dict(IOMT_DEVICE_MAPPING.get(port, DEFAULT_IOMT_CONTEXT))
            device["assumption"] = "Device role inferred from destination port; not part of the dataset."

            explanation = []
            if "top_features" in p and isinstance(p["top_features"], str):
                for feat, ratio in json.loads(p["top_features"]):
                    raw_val = _py(raw[feat]) if raw is not None and feat in raw.index else None
                    explanation.append({"feature": feat, "error_vs_normal": ratio, "raw_value": raw_val})

            score = float(p["anomaly_score"])
            observations = [f"SAFE anomaly score ({self.safe_metadata.get('scorer', 'recon')}) {score:.5f} "
                            f"vs threshold {threshold:.5f} ({score / max(threshold, 1e-12):.2f}x)"]
            for e in explanation[:3]:
                val = "" if e["raw_value"] is None else f" (value {e['raw_value']})"
                observations.append(f"Feature '{e['feature']}' reconstructs {e['error_vs_normal']}x "
                                    f"worse than on normal traffic{val}")

            items.append({
                "evidence_id": f"EVID-{self.dataset_name[:4].upper()}-{i + 1:04d}",
                "flow_id": f"TEST-ROW-{test_row}",
                "timestamp": self._field(raw, "timestamp"),
                "dataset": self.dataset_name,
                "device_context": device,
                "network_flow": flow,
                "detection": {
                    "detector": "SAFE",
                    "scorer": self.safe_metadata.get("scorer", "recon"),
                    "verdict": str(p["verdict"]),
                    "anomaly_score": round(score, 6),
                    "threshold": round(threshold, 6),
                    "anomaly_ratio": round(score / max(threshold, 1e-12), 3),
                },
                "detector_explanation": explanation,
                "observations": observations,
                # evaluation only - stripped before anything is sent to M5's model
                "ground_truth": self._ground_truth(p, raw),
            })

        n_flag = sum(1 for it in items if it["detection"]["verdict"] == "Anomaly")
        print(f"[M4] Built {len(items)} evidence items ({n_flag} flagged by SAFE, {len(items) - n_flag} normal).")
        return items

    def build_and_save_evidence(self, max_samples: int = 100) -> str:
        items = self.load_flagged_flows(max_samples=max_samples)
        payload = {
            "evidence_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": self.dataset_name,
                "total_items": len(items),
                "source_module": "M3-SAFE",
                "detection_threshold": self.safe_metadata.get("threshold"),
                "safe_hyperparameters": {k: self.safe_metadata.get(k) for k in
                                         ["epochs", "mask_ratio", "learning_rate", "hyperparameter_source"]},
                "sampling": f"random (seed {SAMPLE_SEED}): half SAFE-flagged, half SAFE-normal",
                "evidence_schema_version": "2.0",
            },
            "evidence_items": items,
        }
        out_dir = os.path.join(REASONING_RESULTS_DIR, self.dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, "threat_evidence.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, default=str)
        print(f"[M4] Saved threat evidence -> {out_file}")
        return out_file


def main():
    for ds in ["IoMT-TrafficData", "MedSec-25"]:
        print(f"\n=== M4 Threat Evidence Builder: {ds} ===")
        ThreatEvidenceBuilder(ds).build_and_save_evidence(max_samples=100)


if __name__ == "__main__":
    main()