"""
CLI Runner: M6 SDN Mitigation Engine
=====================================
Executes automated SDN flow rule installation and policy enforcement
across IoMT-TrafficData and MedSec-25.

Usage:
  python -m src.mitigation.run_mitigation
  python -m src.mitigation.run_mitigation --dataset IoMT-TrafficData
"""

import os
import sys
import argparse
from datetime import datetime, timezone

from .sdn_controller import SDNMitigationEngine


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

REASONING_DIR = os.path.join(PROJECT_ROOT, "results", "reasoning")


def run_mitigation_pipeline(dataset_name: str, controller_url: str = None):
    print("\n" + "=" * 70)
    print(f"  RUNNING SDN MITIGATION ENGINE (M6): {dataset_name}")
    print("=" * 70)

    reasoning_file = os.path.join(
        REASONING_DIR,
        dataset_name,
        "foundation_sec_reasoning.json"
    )

    if not os.path.exists(reasoning_file):
        print(f"\n[Error] Foundation-Sec reasoning file not found:\n  {reasoning_file}")
        print("Please run M4/M5 reasoning first via:")
        print(f"  python -m src.reasoning.run_reasoning --dataset {dataset_name}")
        return None

    engine = SDNMitigationEngine(
        dataset_name=dataset_name,
        controller_url=controller_url
    )

    summary = engine.enforce_mitigations(reasoning_file)

    print("\n" + "-" * 70)
    print(f"  SDN ENFORCEMENT SUMMARY: {dataset_name}")
    print("-" * 70)
    print(f"Total Directives Processed:     {summary['total_directives_processed']}")
    print(f"OpenFlow 1.3 Rules Installed:   {summary['total_openflow_rules_installed']}")
    print(f"Successful Enforcements:        {summary['successful_enforcements']}")
    print(f"Mitigation Success Rate:        {summary['mitigation_success_rate_percent']:.2f} %")
    print(f"Mean Enforcement Latency / Rule: {summary['mean_enforcement_latency_ms']:.2f} ms")
    print(f"Action Distribution:            {summary['action_breakdown']}")
    print(f"Quarantined Attacker Hosts:     {summary['active_quarantined_hosts']}")
    print("-" * 70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="M6: SDN Mitigation Engine & OpenFlow 1.3 Controller Enforcement"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "IoMT-TrafficData", "MedSec-25"],
        help="Target dataset to run mitigation on (default: all)"
    )
    parser.add_argument(
        "--controller-url",
        type=str,
        default=None,
        help="Optional Ryu/ONOS controller REST endpoint (e.g. http://127.0.0.1:8080)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  AI CYBERSECURITY SOC ASSISTANT")
    print("  M6: SDN Mitigation & OpenFlow Enforcement Layer")
    print("=" * 70)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")

    datasets = (
        ["IoMT-TrafficData", "MedSec-25"]
        if args.dataset == "all"
        else [args.dataset]
    )

    summaries = {}
    for ds in datasets:
        summaries[ds] = run_mitigation_pipeline(
            dataset_name=ds,
            controller_url=args.controller_url
        )

    print("\n" + "=" * 70)
    print("  M6 SDN MITIGATION SUITE COMPLETED")
    print("=" * 70)
    for ds, s in summaries.items():
        if s:
            print(f"  * {ds:<18} -> Success Rate: {s['mitigation_success_rate_percent']}% | Latency: {s['mean_enforcement_latency_ms']} ms")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
