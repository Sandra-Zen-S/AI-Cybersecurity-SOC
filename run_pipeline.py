"""
Master Pipeline Runner: AI Cybersecurity SOC Assistant
======================================================
Executes end-to-end pipeline stages across IoMT-TrafficData and MedSec-25:
  - M4: Threat Evidence Builder
  - M5: Foundation-Sec-8B Cybersecurity Reasoning Layer
  - M6: SDN Automated Mitigation Layer (OpenFlow 1.3)
  - M7: End-to-End Pipeline Evaluation & Benchmarking

Usage:
  python run_pipeline.py --all
  python run_pipeline.py --stage mitigation
  python run_pipeline.py --stage evaluation
  python run_pipeline.py --dataset IoMT-TrafficData
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone

from src.reasoning.run_reasoning import run_pipeline as run_m4_m5
from src.mitigation.run_mitigation import run_mitigation_pipeline as run_m6
from src.evaluation.evaluator import SOCPipelineEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="AI Cybersecurity SOC Assistant: Unified Pipeline Runner"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "reasoning", "mitigation", "evaluation"],
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "IoMT-TrafficData", "MedSec-25"],
        help="Target dataset (default: all)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=25,
        help="Max samples to analyze for reasoning (default: 25)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="offline",
        choices=["offline", "huggingface", "api"],
        help="Foundation-Sec reasoning backend (default: offline)"
    )

    args = parser.parse_args()

    print("=" * 75)
    print("  AI CYBERSECURITY SOC ASSISTANT: MASTER PIPELINE")
    print("  SDN-Enabled IoMT Autonomous Defense & Threat Reasoning")
    print("=" * 75)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Stage:     {args.stage.upper()}")
    print(f"Dataset:   {args.dataset}")
    print(f"Backend:   {args.backend}")

    datasets = (
        ["IoMT-TrafficData", "MedSec-25"]
        if args.dataset == "all"
        else [args.dataset]
    )

    start_master = time.time()

    # --------------------------------------------------------
    # STAGE M4 + M5: Threat Evidence & Foundation-Sec Reasoning
    # --------------------------------------------------------
    if args.stage in ("all", "reasoning"):
        print("\n" + "#" * 75)
        print("  STAGE M4 & M5: THREAT EVIDENCE & FOUNDATION-SEC REASONING")
        print("#" * 75)
        for ds in datasets:
            run_m4_m5(
                dataset_name=ds,
                backend=args.backend,
                max_samples=args.samples
            )

    # --------------------------------------------------------
    # STAGE M6: SDN Automated Mitigation (OpenFlow 1.3)
    # --------------------------------------------------------
    if args.stage in ("all", "mitigation"):
        print("\n" + "#" * 75)
        print("  STAGE M6: SDN AUTOMATED MITIGATION & OPENFLOW ENFORCEMENT")
        print("#" * 75)
        for ds in datasets:
            run_m6(dataset_name=ds)

    # --------------------------------------------------------
    # STAGE M7: End-to-End Evaluation & Benchmarking
    # --------------------------------------------------------
    if args.stage in ("all", "evaluation"):
        print("\n" + "#" * 75)
        print("  STAGE M7: END-TO-END PIPELINE EVALUATION & BENCHMARKING")
        print("#" * 75)
        evaluator = SOCPipelineEvaluator(datasets=datasets)
        output = evaluator.run_full_evaluation()

        comp = evaluator.generate_comparison_matrix(output["results"])
        headers = ["Metric"] + datasets

        print("\n" + "-" * 75)
        print(f"{headers[0]:<32} | {headers[1]:<18} | {headers[2] if len(headers) > 2 else '':<18}")
        print("-" * 75)
        for row in comp.get("comparison_metrics", []):
            m = row.get("metric", "")
            v1 = str(row.get(datasets[0], "N/A"))
            v2 = str(row.get(datasets[1], "N/A")) if len(datasets) > 1 else ""
            print(f"{m:<32} | {v1:<18} | {v2:<18}")
        print("-" * 75)
        print(f"\nEvaluation Report: {output['report']}")

    total_elapsed = time.time() - start_master
    print("\n" + "=" * 75)
    print(f"  ALL REQUESTED STAGES COMPLETED IN {total_elapsed:.2f} s")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
