# AI Cybersecurity SOC Assistant

An AI-driven intrusion detection and response pipeline for SDN-enabled IoMT networks.

## Project Overview

This project investigates an integrated cybersecurity pipeline that combines:

- Whale Optimization Algorithm (WOA) for feature selection
- SAFE for self-supervised anomaly detection
- Foundation-Sec-8B-Instruct for cybersecurity threat reasoning
- MITRE ATT&CK technique mapping
- SDN-based automated mitigation

## Pipeline

IoMT Network Traffic
        ↓
Data Preprocessing
        ↓
WOA Feature Selection
        ↓
SAFE Anomaly Detection
        ↓
Threat Evidence Builder
        ↓
Foundation-Sec-8B-Instruct
        ↓
MITRE ATT&CK Mapping
        ↓
SDN Mitigation
        ↓
Evaluation

## Project Structure

```text
data/           Dataset files
src/            Source code
models/         Trained models
results/        Experiment results
notebooks/      Jupyter notebooks
```

## Running Modules

### M4 & M5: Foundation-Sec Threat Reasoning Layer

To execute the Threat Evidence Builder (M4) and Foundation-Sec-8B reasoning engine (M5):

```bash
# Run full reasoning pipeline across all IoMT datasets
python -m src.reasoning.run_reasoning

# Or run individual stages
python -m src.reasoning.threat_evidence
python -m src.reasoning.foundation_sec --backend offline
```

Output artefacts are saved to `results/reasoning/<dataset_name>/`:
- `threat_evidence.json`: Standardized M4 threat evidence payload.
- `foundation_sec_reasoning.json`: Structured threat analyses with MITRE ATT&CK attribution.
- `mitre_attack_summary.json`: Aggregated ATT&CK tactics, techniques, and SDN directives.
- `threat_report.md`: Human-readable SOC Executive Threat Intelligence report.