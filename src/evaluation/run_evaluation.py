"""
CLI Runner: M7 Comprehensive Pipeline Evaluator
===============================================
Computes benchmark metrics, confusion matrices, latency profiles,
and generates formal evaluation reports across IoMT-TrafficData and MedSec-25.

Usage:
  python -m src.evaluation.run_evaluation
"""

import os
import sys
import argparse
from datetime import datetime, timezone

from .evaluator import SOCPipelineEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="M7: Comprehensive Pipeline Evaluation & Benchmarking Suite"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["IoMT-TrafficData", "MedSec-25"],
        help="Datasets to evaluate (default: IoMT-TrafficData MedSec-25)"
    )

    args = parser.parse_args()

    print("=" * 75)
    print("  AI CYBERSECURITY SOC ASSISTANT")
    print("  M7: End-to-End Pipeline Evaluation & Benchmarking Suite")
    print("=" * 75)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Datasets:  {', '.join(args.datasets)}")

    evaluator = SOCPipelineEvaluator(datasets=args.datasets)
    output = evaluator.run_full_evaluation()

    print("\n" + "=" * 75)
    print("  EVALUATION SUMMARY & CROSS-DATASET BENCHMARKS")
    print("=" * 75)

    comp = evaluator.generate_comparison_matrix(output["results"])
    headers = ["Metric"] + args.datasets

    print(f"{headers[0]:<32} | {headers[1]:<20} | {headers[2]:<20}")
    print("-" * 75)

    for row in comp.get("comparison_metrics", []):
        m = row.get("metric", "")
        v1 = str(row.get(args.datasets[0], "N/A"))
        v2 = str(row.get(args.datasets[1], "N/A"))
        print(f"{m:<32} | {v1:<20} | {v2:<20}")

    print("=" * 75)
    print(f"\n[M7] Evaluation Artefacts Successfully Generated:")
    print(f"  * Metrics Summary:    {output['metrics_summary']}")
    print(f"  * Dataset Comparison: {output['comparison']}")
    print(f"  * Executive Report:   {output['report']}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
