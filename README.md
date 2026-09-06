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