"""
Pipeline Runner: M4 Evidence Builder + M5 Foundation-Sec Reasoning
==================================================================
Runs the end-to-end Threat Evidence Building (M4) and Foundation-Sec
Reasoning (M5) pipeline across IoMT-TrafficData and MedSec-25.

Outputs:
  - results/reasoning/<dataset>/threat_evidence.json
  - results/reasoning/<dataset>/foundation_sec_reasoning.json
  - results/reasoning/<dataset>/mitre_attack_summary.json
  - results/reasoning/<dataset>/threat_report.md
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime, timezone

from .threat_evidence import ThreatEvidenceBuilder
from .foundation_sec import FoundationSecReasoner


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

REASONING_DIR = os.path.join(PROJECT_ROOT, "results", "reasoning")


# ============================================================
# RUN REASONING PIPELINE
# ============================================================

def run_pipeline(
    dataset_name: str,
    backend: str = "offline",
    max_samples: int = 25,
    model_id: str = "cisco/Foundation-Sec-8B-Instruct"
):
    print("\n" + "=" * 70)
    print(f"  RUNNING FOUNDATION-SEC PIPELINE: {dataset_name}")
    print("=" * 70)

    start_total = time.time()

    # --------------------------------------------------------
    # Stage M4: Threat Evidence Builder
    # --------------------------------------------------------
    print(f"\n[Stage M4] Building structured threat evidence from SAFE detections...")
    start_m4 = time.time()
    evidence_builder = ThreatEvidenceBuilder(dataset_name)
    evidence_file = evidence_builder.build_and_save_evidence(max_samples=max_samples)
    elapsed_m4 = time.time() - start_m4

    print(f"[Stage M4] Complete in {elapsed_m4:.3f}s -> {os.path.basename(evidence_file)}")

    # --------------------------------------------------------
    # Stage M5: Foundation-Sec-8B Reasoning Layer
    # --------------------------------------------------------
    print(f"\n[Stage M5] Applying Foundation-Sec-8B-Instruct reasoning (backend: '{backend}')...")
    start_m5 = time.time()
    reasoner = FoundationSecReasoner(
        dataset_name=dataset_name,
        backend=backend,
        model_id=model_id
    )
    analyses = reasoner.analyze_evidence(evidence_file)
    saved_artefacts = reasoner.save_results(analyses)
    elapsed_m5 = time.time() - start_m5

    total_time = time.time() - start_total
    avg_latency_ms = (elapsed_m5 / max(1, len(analyses))) * 1000

    # --------------------------------------------------------
    # Pipeline Summary Metrics
    # --------------------------------------------------------
    print("\n" + "-" * 70)
    print(f"  PERFORMANCE & REASONING SUMMARY: {dataset_name}")
    print("-" * 70)
    attack_count = sum(1 for a in analyses if a.get("is_attack", True))
    normal_count = sum(1 for a in analyses if not a.get("is_attack", True))

    print(f"Total Flows Reasoned:              {len(analyses)}")
    print(f"  - Malicious Attack Flows:        {attack_count}")
    print(f"  - Benign / Normal Clinical Flows: {normal_count}")
    print(f"M4 Evidence Builder Latency:       {elapsed_m4:.3f} s")
    print(f"M5 Foundation-Sec Reasoning Time:  {elapsed_m5:.3f} s")
    print(f"Average Inference Latency / Flow:  {avg_latency_ms:.2f} ms")
    print(f"End-to-End Pipeline Latency:       {total_time:.3f} s")

    # Load summary json for verification
    with open(saved_artefacts["summary"], "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    print("\n[MITRE ATT&CK Attribution Coverage]")
    for tech, count in summary_data["mitre_techniques_distribution"].items():
        print(f"  - {tech:<40}: {count} incident(s)")

    print("\n[Automated SDN Mitigation Directives]")
    for act, count in summary_data["sdn_mitigation_actions"].items():
        print(f"  - Action `{act}`: {count} flow rule(s) ready for OpenFlow controller")

    print("-" * 70)
    return {
        "dataset": dataset_name,
        "total_analyzed": len(analyses),
        "elapsed_m4_sec": elapsed_m4,
        "elapsed_m5_sec": elapsed_m5,
        "avg_latency_ms": avg_latency_ms,
        "saved_artefacts": saved_artefacts
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AI SOC Assistant - Foundation-Sec Reasoning Pipeline")
    parser.add_argument("--dataset", choices=["IoMT-TrafficData", "MedSec-25", "all"], default="all")
    parser.add_argument("--backend", choices=["offline", "huggingface", "api"], default="offline")
    parser.add_argument("--samples", type=int, default=25, help="Number of evidence flows to reason over")
    parser.add_argument("--model-id", default="cisco/Foundation-Sec-8B-Instruct")
    args = parser.parse_args()

    print("=" * 70)
    print("  AI CYBERSECURITY SOC ASSISTANT")
    print("  M4/M5: Threat Evidence Builder & Foundation-Sec Reasoning Layer")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Backend:   {args.backend.upper()}")
    print(f"Model ID:  {args.model_id}")

    datasets = ["IoMT-TrafficData", "MedSec-25"] if args.dataset == "all" else [args.dataset]
    results = []

    for ds in datasets:
        res = run_pipeline(
            dataset_name=ds,
            backend=args.backend,
            max_samples=args.samples,
            model_id=args.model_id
        )
        results.append(res)

    print("\n" + "=" * 70)
    print("  ALL DATASETS PROCESSED SUCCESSFULLY BY FOUNDATION-SEC!")
    print("=" * 70)
    for r in results:
        print(f"  Dataset: {r['dataset']:<20} | Flagged Flows: {r['total_analyzed']} | Avg Latency: {r['avg_latency_ms']:.2f} ms")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
