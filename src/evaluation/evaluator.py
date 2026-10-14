"""
M7 - Evaluation Engine
======================
Collects MEASURED metrics from every pipeline stage. Nothing is hard-coded:
if a stage has not been run, its values are reported as N/A.

Inputs
  - results/woa/<dataset>/selected_features.json
  - results/safe/<dataset>/safe_metadata.json, safe_predictions.csv
  - results/reasoning/<dataset>/mitre_attack_summary.json (incl. evaluation)
  - results/reasoning/<dataset>/foundation_sec_reasoning.json
  - results/reasoning/<dataset>/pipeline_timing.json
  - results/mitigation/<dataset>/mitigation_summary.json (optional)

Outputs
  - results/evaluation/metrics_summary.json
  - results/evaluation/cross_dataset_comparison.json
  - results/evaluation/evaluation_report.md
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WOA_DIR = os.path.join(PROJECT_ROOT, "results", "woa")
SAFE_DIR = os.path.join(PROJECT_ROOT, "results", "safe")
REASONING_DIR = os.path.join(PROJECT_ROOT, "results", "reasoning")
MITIGATION_DIR = os.path.join(PROJECT_ROOT, "results", "mitigation")
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "results", "evaluation")


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def pct(x):
    return None if x is None else round(100.0 * float(x), 2)


def show(v, suffix=""):
    return "N/A" if v is None else f"{v}{suffix}"


class SOCPipelineEvaluator:

    def __init__(self, datasets: List[str] = None):
        self.datasets = datasets or ["IoMT-TrafficData", "MedSec-25"]
        self.output_dir = EVALUATION_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------------
    # M2 - WOA
    # --------------------------------------------------------

    def evaluate_m2_woa(self, dataset: str) -> Dict[str, Any]:
        data = load_json(os.path.join(WOA_DIR, dataset, "selected_features.json"))
        if not data:
            return {"available": False}
        total = data.get("total_features")
        selected = data.get("selected_feature_count", len(data.get("selected_features", [])))
        return {
            "available": True,
            "total_raw_features": total,
            "woa_selected_features": selected,
            "feature_reduction_percentage": round(100.0 * (total - selected) / total, 2) if total else None,
        }

    # --------------------------------------------------------
    # M3 - SAFE
    # --------------------------------------------------------

    def evaluate_m3_safe(self, dataset: str) -> Dict[str, Any]:
        meta = load_json(os.path.join(SAFE_DIR, dataset, "safe_metadata.json"))
        pred_path = os.path.join(SAFE_DIR, dataset, "safe_predictions.csv")
        if not meta or not os.path.exists(pred_path):
            return {"available": False}

        p = pd.read_csv(pred_path)
        y, yhat, s = p["true_binary_label"].values, p["predicted_binary_label"].values, p["anomaly_score"].values
        tp = int(((yhat == 1) & (y == 1)).sum())
        fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum())
        fn = int(((yhat == 0) & (y == 1)).sum())
        mean_attack = float(np.mean(s[y == 1])) if (y == 1).any() else None
        mean_normal = float(np.mean(s[y == 0])) if (y == 0).any() else None

        return {
            "available": True,
            "test_samples": int(len(y)),
            "attack_share_percent": round(100.0 * (y == 1).mean(), 2),
            "confusion_matrix": {"true_positives": tp, "false_positives": fp,
                                 "true_negatives": tn, "false_negatives": fn},
            "accuracy_percent": pct(meta.get("accuracy")),
            "precision_percent": pct(meta.get("precision")),
            "recall_detection_rate_percent": pct(meta.get("recall")),
            "f1_score_percent": pct(meta.get("f1_score")),
            "specificity_percent": round(100.0 * tn / max(tn + fp, 1), 2),
            "false_positive_rate_percent": round(100.0 * fp / max(fp + tn, 1), 2),
            "balanced_accuracy_percent": pct(meta.get("balanced_accuracy")),
            "mcc": round(float(meta["mcc"]), 4) if meta.get("mcc") is not None else None,
            "scorer": meta.get("scorer", "recon"),
            "baseline_all_attack_f1_percent": pct(meta.get("baseline_all_attack_f1")),
            "roc_auc": round(float(meta["roc_auc"]), 4) if meta.get("roc_auc") is not None else None,
            "pr_auc": round(float(meta["pr_auc"]), 4) if meta.get("pr_auc") is not None else None,
            "threshold": meta.get("threshold"),
            "mean_score_attack": round(mean_attack, 6) if mean_attack is not None else None,
            "mean_score_normal": round(mean_normal, 6) if mean_normal is not None else None,
            "score_separation_ratio": round(mean_attack / max(mean_normal, 1e-12), 2)
            if mean_attack is not None and mean_normal is not None else None,
            "hyperparameters": {k: meta.get(k) for k in
                                ["epochs", "mask_ratio", "learning_rate", "batch_size", "latent_dim",
                                 "dropout", "loss", "hyperparameter_source"]},
        }

    # --------------------------------------------------------
    # M4/M5 - Evidence + reasoning
    # --------------------------------------------------------

    def evaluate_m5_reasoning(self, dataset: str) -> Dict[str, Any]:
        summary = load_json(os.path.join(REASONING_DIR, dataset, "mitre_attack_summary.json"))
        timing = load_json(os.path.join(REASONING_DIR, dataset, "pipeline_timing.json")) or {}
        if not summary or "evaluation" not in summary:
            return {"available": False}
        ev = summary["evaluation"]
        return {
            "available": True,
            "backend": summary.get("reasoning_backend"),
            **ev,
            "techniques_distribution": summary.get("mitre_techniques_distribution", {}),
            "severity_breakdown": summary.get("severity_distribution", {}),
            "action_breakdown": summary.get("sdn_mitigation_actions", {}),
            "m4_evidence_builder_s": timing.get("m4_evidence_builder_s"),
            "m5_reasoning_s": timing.get("m5_reasoning_s"),
            "m5_mean_latency_ms_per_flow": timing.get("m5_mean_latency_ms_per_flow"),
        }

    # --------------------------------------------------------
    # M6 - Mitigation (only what the mitigation module recorded)
    # --------------------------------------------------------

    def evaluate_m6_mitigation(self, dataset: str) -> Dict[str, Any]:
        summary = load_json(os.path.join(MITIGATION_DIR, dataset, "mitigation_summary.json"))
        if not summary:
            return {"available": False}
        return {
            "available": True,
            "openflow_rules_installed": summary.get("total_openflow_rules_installed"),
            "successful_enforcements": summary.get("successful_enforcements"),
            "enforcement_success_rate_percent": summary.get("mitigation_success_rate_percent"),
            "mean_enforcement_latency_ms": summary.get("mean_enforcement_latency_ms"),
            "action_breakdown": summary.get("action_breakdown", {}),
        }

    # --------------------------------------------------------
    # Per dataset
    # --------------------------------------------------------

    def evaluate_end_to_end_pipeline(self, dataset: str) -> Dict[str, Any]:
        m2, m3 = self.evaluate_m2_woa(dataset), self.evaluate_m3_safe(dataset)
        m5, m6 = self.evaluate_m5_reasoning(dataset), self.evaluate_m6_mitigation(dataset)

        per_flow = None
        if m5.get("available") and m5.get("m4_evidence_builder_s") is not None and m5.get("flows_analysed"):
            per_flow = (m5["m4_evidence_builder_s"] * 1000.0 / m5["flows_analysed"]
                        + (m5.get("m5_mean_latency_ms_per_flow") or 0.0)
                        + (m6.get("mean_enforcement_latency_ms") or 0.0))
        return {
            "dataset": dataset,
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "woa_feature_selection": m2,
            "safe_anomaly_detection": m3,
            "threat_reasoning": m5,
            "sdn_mitigation": m6,
            "latency_profile": {
                "m4_evidence_builder_s": m5.get("m4_evidence_builder_s"),
                "m5_reasoning_s": m5.get("m5_reasoning_s"),
                "m6_mean_enforcement_latency_ms": m6.get("mean_enforcement_latency_ms"),
                "per_flow_m4_to_m6_ms": round(per_flow, 3) if per_flow is not None else None,
                "note": "M4 time includes linking flows to raw records; SAFE inference time is reported by compare_models.py.",
            },
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        results = {ds: self.evaluate_end_to_end_pipeline(ds) for ds in self.datasets}
        comparison = self.generate_comparison_matrix(results)

        summary_file = os.path.join(self.output_dir, "metrics_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        comparison_file = os.path.join(self.output_dir, "cross_dataset_comparison.json")
        with open(comparison_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        report_file = os.path.join(self.output_dir, "evaluation_report.md")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(self.generate_markdown_report(results, comparison))
        return {"metrics_summary": summary_file, "comparison": comparison_file,
                "report": report_file, "results": results}

    def generate_comparison_matrix(self, results: Dict[str, Any]) -> Dict[str, Any]:
        rows = [
            ("Raw features", lambda r: r["woa_feature_selection"].get("total_raw_features"), ""),
            ("WOA selected features", lambda r: r["woa_feature_selection"].get("woa_selected_features"), ""),
            ("Feature reduction", lambda r: r["woa_feature_selection"].get("feature_reduction_percentage"), "%"),
            ("SAFE ROC-AUC", lambda r: r["safe_anomaly_detection"].get("roc_auc"), ""),
            ("SAFE PR-AUC", lambda r: r["safe_anomaly_detection"].get("pr_auc"), ""),
            ("SAFE scorer", lambda r: r["safe_anomaly_detection"].get("scorer"), ""),
            ("SAFE balanced accuracy", lambda r: r["safe_anomaly_detection"].get("balanced_accuracy_percent"), "%"),
            ("SAFE MCC", lambda r: r["safe_anomaly_detection"].get("mcc"), ""),
            ("SAFE F1", lambda r: r["safe_anomaly_detection"].get("f1_score_percent"), "%"),
            ("Flag-everything baseline F1", lambda r: r["safe_anomaly_detection"].get("baseline_all_attack_f1_percent"), "%"),
            ("SAFE precision", lambda r: r["safe_anomaly_detection"].get("precision_percent"), "%"),
            ("SAFE recall", lambda r: r["safe_anomaly_detection"].get("recall_detection_rate_percent"), "%"),
            ("SAFE false-positive rate", lambda r: r["safe_anomaly_detection"].get("false_positive_rate_percent"), "%"),
            ("SAFE accuracy", lambda r: r["safe_anomaly_detection"].get("accuracy_percent"), "%"),
            ("Attack share in test set", lambda r: r["safe_anomaly_detection"].get("attack_share_percent"), "%"),
            ("ATT&CK attribution accuracy", lambda r: r["threat_reasoning"].get("attribution_accuracy_percent"), "%"),
            ("Attack mitigation coverage", lambda r: r["threat_reasoning"].get("attack_mitigation_coverage_percent"), "%"),
            ("False-mitigation rate", lambda r: r["threat_reasoning"].get("false_mitigation_rate_percent"), "%"),
            ("Reasoning backend", lambda r: r["threat_reasoning"].get("backend"), ""),
            ("Per-flow M4-M6 latency", lambda r: r["latency_profile"].get("per_flow_m4_to_m6_ms"), " ms"),
        ]
        table = []
        for name, getter, suffix in rows:
            row = {"metric": name}
            for ds in self.datasets:
                if ds in results:
                    try:
                        row[ds] = show(getter(results[ds]), suffix)
                    except (KeyError, TypeError):
                        row[ds] = "N/A"
            table.append(row)
        return {"comparison_metrics": table}

    def generate_markdown_report(self, results: Dict[str, Any], comparison: Dict[str, Any]) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        dss = [d for d in self.datasets if d in results]
        md = [
            "# AI Cybersecurity SOC Assistant - Evaluation Report",
            f"**Generated:** `{now}`  ",
            "**Pipeline:** WOA feature selection (M2) -> SAFE anomaly detection (M3) -> threat evidence (M4) "
            "-> threat reasoning (M5) -> SDN mitigation (M6)  ",
            "All values below are read from the stage outputs; N/A means the stage was not run.", "",
            "## 1. Cross-dataset summary", "",
            "| Metric | " + " | ".join(dss) + " |",
            "| :--- | " + " | ".join([":---:"] * len(dss)) + " |",
        ]
        for row in comparison["comparison_metrics"]:
            md.append(f"| {row['metric']} | " + " | ".join(str(row.get(d, "N/A")) for d in dss) + " |")

        for ds in dss:
            r = results[ds]
            m3, m5 = r["safe_anomaly_detection"], r["threat_reasoning"]
            md += ["", f"## {ds}", ""]
            if m3.get("available"):
                cm = m3["confusion_matrix"]
                md += ["### SAFE detection", "",
                       f"- Test samples: {m3['test_samples']} ({m3['attack_share_percent']}% attacks)",
                       f"- Confusion matrix: TP {cm['true_positives']}, FP {cm['false_positives']}, "
                       f"TN {cm['true_negatives']}, FN {cm['false_negatives']}",
                       f"- ROC-AUC {show(m3['roc_auc'])}, PR-AUC {show(m3['pr_auc'])}, F1 {show(m3['f1_score_percent'], '%')}, "
                       f"precision {show(m3['precision_percent'], '%')}, recall {show(m3['recall_detection_rate_percent'], '%')}, "
                       f"FPR {show(m3['false_positive_rate_percent'], '%')}, "
                       f"balanced accuracy {show(m3['balanced_accuracy_percent'], '%')}, MCC {show(m3['mcc'])} "
                       f"(scorer: {m3['scorer']})",
                       f"- Mean anomaly score: attack {show(m3['mean_score_attack'])}, normal {show(m3['mean_score_normal'])} "
                       f"(ratio {show(m3['score_separation_ratio'], 'x')})",
                       f"- Hyperparameters: `{m3['hyperparameters']}`",
                       "- Note: accuracy is dominated by the attack share of the test set; ROC-AUC, PR-AUC and F1 are more informative."]
            if m5.get("available"):
                md += ["", "### Threat reasoning", "",
                       f"- Backend: {m5['backend']} ({m5['flows_analysed']} sampled flows: "
                       f"{m5['ground_truth_attacks']} attacks, {m5['ground_truth_benign']} benign by ground truth)",
                       f"- ATT&CK attribution accuracy: {show(m5['attribution_accuracy_percent'], '%')} "
                       f"(technique level, {m5['attribution_scored_flows']} flagged attacks scored)",
                       f"- Attack mitigation coverage: {show(m5['attack_mitigation_coverage_percent'], '%')}; "
                       f"missed attacks: {show(m5['missed_attack_rate_percent'], '%')}",
                       f"- False-mitigation rate (benign flows restricted): {show(m5['false_mitigation_rate_percent'], '%')}",
                       f"- Valid model output: {show(m5['llm_valid_output_percent'], '%')}; "
                       f"mean latency {show(m5['mean_reasoning_latency_ms'], ' ms')}"]
                if m5["backend"] == "offline":
                    md.append("- The offline backend is a rule-based engine, not a language model.")
        return "\n".join(md) + "\n"