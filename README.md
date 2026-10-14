# AI Cybersecurity SOC Assistant

An AI-driven intrusion detection, threat reasoning, and automated mitigation pipeline for SDN-enabled Internet of Medical Things (IoMT) networks.

## Project Overview

This project investigates an integrated cybersecurity pipeline that combines:

- **Whale Optimization Algorithm (WOA)** for metaheuristic feature selection
- **SAFE** (Self-Supervised Masked Autoencoder) for label-efficient anomaly detection
- **Foundation-Sec-8B-Instruct** for domain-specific threat reasoning and clinical impact assessment
- **MITRE ATT&CK Matrix** technique and tactic attribution
- **SDN Automated Mitigation** (OpenFlow 1.3 flow table modification rules)
- **Comprehensive Benchmarking & Evaluation** across clinical benchmarks (`IoMT-TrafficData` and `MedSec-25`)

---

## Pipeline Architecture

```text
IoMT Clinical Network Traffic (Infusion Pumps, PACS, EHR, Telemetry)
                        ↓
             M1: Data Preprocessing
                        ↓
            M2: WOA Feature Selection (45-56% Dimensionality Reduction)
                        ↓
            M3: SAFE Self-Supervised Anomaly Detection (MSE Anomaly Scoring)
                        ↓
            M4: Threat Evidence Builder (Standardized Telemetry JSON)
                        ↓
            M5: Foundation-Sec-8B Threat Reasoning (ATT&CK Attribution)
                        ↓
            M6: SDN Automated Mitigation (OpenFlow 1.3 Flow Rules)
                        ↓
            M7: End-to-End Pipeline Evaluation & Benchmarking
```

---

## Project Structure

```text
AI-Cybersecurity-SOC/
├── data/
│   └── processed/          # Preprocessed datasets (IoMT-TrafficData, MedSec-25)
├── src/
│   ├── preprocessing/      # M1: Ingestion, cleaning, encoding, normalization
│   ├── optimization/       # M2: Whale Optimization Algorithm (WOA)
│   ├── safe/               # M3: Self-supervised Masked Autoencoder (SAFE)
│   ├── reasoning/          # M4/M5: Threat Evidence Builder & Foundation-Sec-8B
│   ├── mitigation/         # M6: OpenFlow 1.3 SDN Mitigation Engine & Switch Emulator
│   └── evaluation/         # M7: Metrics calculator & benchmark report generator
├── results/
│   ├── woa/                # Selected feature indices & fitness logs
│   ├── reasoning/          # Evidence objects, Foundation-Sec analyses, threat reports
│   ├── mitigation/         # OpenFlow 1.3 rules, controller states, mitigation logs
│   └── evaluation/         # Metrics summary, cross-dataset comparison, formal report
└── run_pipeline.py         # Master unified end-to-end pipeline runner
```

---

## Running the Pipeline

### 1. Master Pipeline Runner (All Modules)

To run the complete pipeline end-to-end across all datasets:

```bash
# Execute Reasoning (M4/M5) -> Mitigation (M6) -> Evaluation (M7)
python run_pipeline.py --all

# Or run a specific stage
python run_pipeline.py --stage mitigation
python run_pipeline.py --stage evaluation
```

---

### 2. M4 & M5: Foundation-Sec Threat Reasoning Layer

To execute the Threat Evidence Builder (M4) and Foundation-Sec-8B reasoning engine (M5):

```bash
# Run full reasoning pipeline across all IoMT datasets
python -m src.reasoning.run_reasoning

# Or run with specific backend (offline, huggingface, api)
python -m src.reasoning.run_reasoning --dataset IoMT-TrafficData --backend offline
```

**Output Artefacts (`results/reasoning/<dataset>/`):**
- `threat_evidence.json`: Standardized M4 threat evidence payload.
- `foundation_sec_reasoning.json`: Structured threat analyses with MITRE ATT&CK attribution.
- `mitre_attack_summary.json`: Aggregated ATT&CK tactics, techniques, and SDN directives.
- `threat_report.md`: Human-readable SOC Executive Threat Intelligence report.

---

### 3. M6: SDN Automated Mitigation Layer (OpenFlow 1.3)

To translate Foundation-Sec directives into concrete OpenFlow 1.3 switch rules and enforce them:

```bash
# Run mitigation across both datasets
python -m src.mitigation.run_mitigation

# Run on a single dataset
python -m src.mitigation.run_mitigation --dataset IoMT-TrafficData
```

**Directives Enforced:**
- `BLOCK_FLOW`: OpenFlow DROP rule matching 5-tuple (`priority=60000`, `idle_timeout=600`, `hard_timeout=7200`)
- `QUARANTINE_HOST`: OpenFlow `SET_FIELD vlan_vid=99` to isolate attacker IP into containment sink (`priority=65000`)
- `ISOLATE_PORT`: Ingress port security shutdown on switch interface (`priority=65500`)
- `ALLOW_FLOW`: OpenFlow normal forwarding for benign clinical telemetry (`priority=1000`)

**Output Artefacts (`results/mitigation/<dataset>/`):**
- `sdn_flow_rules.json`: OpenFlow 1.3 JSON flow table modifications.
- `mitigation_log.json`: Per-evidence mitigation audit log with enforcement timestamps and latencies.
- `sdn_controller_state.json`: Active flow tables, port statistics, and TCAM utilization.
- `mitigation_summary.json`: Aggregated mitigation success rate (%) and mean enforcement latency (ms).

---

### 4. M7: End-to-End Evaluation & Benchmarking

To compute comprehensive evaluation metrics, confusion matrices, and generate the academic evaluation report:

```bash
python -m src.evaluation.run_evaluation
```

**Output Artefacts (`results/evaluation/`):**
- `metrics_summary.json`: Machine-readable quantitative evaluation metrics for all stages.
- `cross_dataset_comparison.json`: Side-by-side benchmarking matrix (`IoMT-TrafficData` vs `MedSec-25`).
- `evaluation_report.md`: Publication-grade technical report with evaluation tables, methodology, and clinical safety assessment.