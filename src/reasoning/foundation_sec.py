"""
M5 - Foundation-Sec Reasoning Layer
====================================
Turns M4 threat evidence (real SAFE detections) into structured threat
analyses: MITRE ATT&CK attribution, clinical impact and an SDN mitigation
directive.

Inputs
  - results/reasoning/<dataset>/threat_evidence.json  (from M4)

Outputs
  - results/reasoning/<dataset>/foundation_sec_reasoning.json
  - results/reasoning/<dataset>/mitre_attack_summary.json  (incl. evaluation)
  - results/reasoning/<dataset>/threat_report.md

Backends
  1. offline (default)  RULE-BASED engine (not a language model). Benign vs
                        attack follows the SAFE verdict; the attack type comes
                        from simple evidence rules (port / protocol / flags).
                        Flows that match no rule are reported as unclassified.
  2. huggingface        Local Foundation-Sec-8B-Instruct via transformers
                        (needs a GPU; ~16 GB in bf16, or set FOUNDATION_SEC_4BIT=1
                        with bitsandbytes for ~6 GB).
  3. api                Any OpenAI-compatible chat endpoint (Ollama, vLLM,
                        LM Studio). Set FOUNDATION_SEC_API_URL, e.g.
                        http://localhost:11434/v1/chat/completions, and
                        --model-id to the served model name.

The reasoning step NEVER sees the ground truth. Ground truth is attached to
each analysis only afterwards, to compute evaluation metrics:
  - attribution accuracy (technique level) on attacks SAFE flagged
  - false-mitigation rate (benign flows given a blocking/limiting action)
  - missed-attack rate (attack flows given ALLOW_FLOW)
"""

import os
import re
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REASONING_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "reasoning")


# ============================================================
# PROMPT
# ============================================================

FOUNDATION_SEC_SYSTEM_PROMPT = """You are a cybersecurity analyst model specialised in threat intelligence, MITRE ATT&CK attribution and Internet of Medical Things (IoMT) network defence.

You receive one network flow scored by the SAFE self-supervised anomaly detector, with the detector's verdict, its most anomalous features and the flow record. Some fields may be null when the dataset does not contain them.

Tasks:
1. Decide whether the evidence indicates malicious activity. Consider the detector verdict, but the detector can be wrong.
2. If malicious, attribute it to the most specific MITRE ATT&CK tactic and technique the evidence supports. If it cannot be attributed, use technique_id "None".
3. Assess the clinical / patient-safety impact.
4. Recommend one SDN-enforceable action. Use ALLOW_FLOW for benign traffic.

Answer with ONE valid JSON object that follows the schema exactly, and nothing else."""

OUTPUT_SCHEMA = """{
  "threat_summary": "<1-3 sentences>",
  "threat_type": "<classification, or 'Benign' >",
  "severity": "INFORMATIONAL | LOW | MEDIUM | HIGH | CRITICAL",
  "confidence_score": <float 0.0-1.0>,
  "mitre_attack": {
    "tactic": "<tactic name or 'None (Benign)'>",
    "tactic_id": "<TAxxxx or 'N/A'>",
    "technique_id": "<Txxxx, Txxxx.xxx or 'None'>",
    "technique_name": "<name>",
    "sub_technique_id": "<Txxxx.xxx or null>",
    "sub_technique_name": "<name or null>",
    "description": "<why this technique>"
  },
  "cwe_cve_references": [{"id": "<CWE/CVE id>", "description": "<text>"}],
  "clinical_impact_assessment": {
    "patient_safety_risk": "NEGLIGIBLE | LOW | MODERATE | HIGH | CRITICAL",
    "clinical_service_disruption": "<text>",
    "data_integrity_loss": "<text>"
  },
  "mitigation_directive": {
    "action": "ALLOW_FLOW | RATE_LIMIT | BLOCK_FLOW | QUARANTINE_HOST | ISOLATE_PORT",
    "priority": <int 1-5>,
    "sdn_action": "FORWARD | METER_POLICING | DROP | SET_VLAN_QUARANTINE | PORT_SECURITY_SHUTDOWN",
    "parameters": {"rate_limit_pps": <int>, "hard_timeout_sec": <int>, "idle_timeout_sec": <int>, "vlan_quarantine": <bool>},
    "technical_rationale": "<text>",
    "clinical_rationale": "<text>"
  }
}"""

ALLOWED_ACTIONS = {"ALLOW_FLOW", "RATE_LIMIT", "BLOCK_FLOW", "QUARANTINE_HOST", "ISOLATE_PORT"}
ALLOWED_SEVERITY = {"INFORMATIONAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
TECHNIQUE_RE = re.compile(r"^T\d{4}(\.\d{3})?$")
BENIGN_LABELS = {"normal", "benign"}


# ============================================================
# MITRE ATT&CK & IOMT CYBERSECURITY KNOWLEDGE BASE
# (used by the rule-based engine and as ground-truth reference for
#  scoring attribution: true attack label -> expected technique)
# ============================================================

IOMT_ATTACK_KNOWLEDGE_BASE = {
    # --------------------------------------------------------
    # IoMT-TrafficData Vectors
    # --------------------------------------------------------
    "apachekiller": {
        "threat_summary": "HTTP Range header byte-range amplification Denial-of-Service attack (Apache Killer) targeting clinical web management interface, driving memory exhaustion and crashing patient portal worker threads.",
        "threat_type": "Application Layer Denial of Service (Range Header Amplification)",
        "severity": "CRITICAL",
        "confidence_score": 0.96,
        "mitre_attack": {
            "tactic": "Impact",
            "tactic_id": "TA0040",
            "technique_id": "T1499",
            "technique_name": "Endpoint Denial of Service",
            "sub_technique_id": "T1499.002",
            "sub_technique_name": "Service Exhaustion",
            "description": "Adversary sends malicious HTTP requests with numerous overlapping byte-range headers to exhaust server memory pools and process threads."
        },
        "cwe_cve_references": [
            {"id": "CVE-2011-3192", "description": "Apache HTTP Server Range Header Byte-Range DoS"},
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "CRITICAL",
            "clinical_service_disruption": "Crashing of medical web gateway prevents clinical staff from adjusting infusion parameters or reviewing diagnostic radiology feeds.",
            "data_integrity_loss": "Low direct telemetry corruption, but complete loss of clinical telemetry accessibility."
        },
        "mitigation_directive": {
            "action": "BLOCK_FLOW",
            "priority": 1,
            "sdn_action": "DROP",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 7200,
                "idle_timeout_sec": 600,
                "vlan_quarantine": False
            },
            "technical_rationale": "Install high-priority SDN flow rule matching ingress switch port and attacker IPv4 to immediately drop TCP traffic destined for HTTP port 80.",
            "clinical_rationale": "Preserves clinical web gateway resources, preventing denial of life-critical infusion telemetry services."
        }
    },

    "arpspoofing": {
        "threat_summary": "Man-in-the-Middle (MitM) ARP cache poisoning intercepting unencrypted L2 communications between medical monitoring stations and clinical telemetry collectors.",
        "threat_type": "Adversary-in-the-Middle (ARP Cache Poisoning)",
        "severity": "CRITICAL",
        "confidence_score": 0.94,
        "mitre_attack": {
            "tactic": "Credential Access / Collection",
            "tactic_id": "TA0006",
            "technique_id": "T1557",
            "technique_name": "Adversary-in-the-Middle",
            "sub_technique_id": "T1557.002",
            "sub_technique_name": "ARP Poisoning",
            "description": "Adversary crafts gratuitous or forged ARP replies to link their MAC address with a clinical gateway IP, eavesdropping or tampering with patient flows."
        },
        "cwe_cve_references": [
            {"id": "CWE-300", "description": "Channel Accessible by Non-Endpoint"},
            {"id": "CWE-319", "description": "Cleartext Transmission of Sensitive Information"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "CRITICAL",
            "clinical_service_disruption": "Adversary can alter patient vital sign streams or inject spoofed biometric alarms, leading to incorrect medication or treatment delays.",
            "data_integrity_loss": "High risk of patient physiological data tampering in transit."
        },
        "mitigation_directive": {
            "action": "QUARANTINE_HOST",
            "priority": 1,
            "sdn_action": "SET_VLAN_QUARANTINE",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 14400,
                "idle_timeout_sec": 1200,
                "vlan_quarantine_id": 999
            },
            "technical_rationale": "SDN controller dynamically pushes OpenFlow flow modification remapping offending MAC/port to an isolated remediation VLAN with Dynamic ARP Inspection (DAI).",
            "clinical_rationale": "Neutralizes Man-in-the-Middle position immediately while preventing packet corruption across the clinical patient monitoring network."
        }
    },

    "mqttmalaria": {
        "threat_summary": "High-frequency MQTT broker volumetric flooding attack saturating IoT telemetry queues for pulse oximeters, smart beds, and cardiac monitoring sensors.",
        "threat_type": "Network Denial of Service (MQTT Telemetry Flood)",
        "severity": "HIGH",
        "confidence_score": 0.93,
        "mitre_attack": {
            "tactic": "Impact",
            "tactic_id": "TA0040",
            "technique_id": "T1498",
            "technique_name": "Network Denial of Service",
            "sub_technique_id": "T1498.001",
            "sub_technique_name": "Direct Network Flood",
            "description": "Adversary floods the medical MQTT broker with bogus publish/subscribe requests, exhausting broker RAM and dropping physiological sensor frames."
        },
        "cwe_cve_references": [
            {"id": "CWE-770", "description": "Allocation of Resources Without Limits or Throttling"},
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "HIGH",
            "clinical_service_disruption": "Delayed or dropped critical cardiac event telemetry from ICU patient vitals monitors.",
            "data_integrity_loss": "High packet loss across continuous physiological telemetry streams."
        },
        "mitigation_directive": {
            "action": "RATE_LIMIT",
            "priority": 2,
            "sdn_action": "METER_POLICING",
            "parameters": {
                "rate_limit_pps": 50,
                "hard_timeout_sec": 3600,
                "idle_timeout_sec": 300,
                "vlan_quarantine": False
            },
            "technical_rationale": "Program OpenFlow meter band policing traffic on TCP port 1883/8883 from offending source IP, capping bandwidth to 50 pps.",
            "clinical_rationale": "Throttles the attack flood while allowing legitimate low-bandwidth medical sensor readings to reach the clinical broker."
        }
    },

    "slowloris": {
        "threat_summary": "Slowloris application starvation attack keeping hundreds of HTTP connections partially open against a hospital clinical portal, depleting available socket descriptors.",
        "threat_type": "Application Layer Denial of Service (Slow HTTP Headers)",
        "severity": "HIGH",
        "confidence_score": 0.95,
        "mitre_attack": {
            "tactic": "Impact",
            "tactic_id": "TA0040",
            "technique_id": "T1499",
            "technique_name": "Endpoint Denial of Service",
            "sub_technique_id": "T1499.003",
            "sub_technique_name": "Application Exhaustion Flood",
            "description": "Adversary continuously sends fragmented HTTP headers at slow intervals, tying up web server connections indefinitely."
        },
        "cwe_cve_references": [
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"},
            {"id": "CWE-799", "description": "Improper Control of Interaction Frequency"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "HIGH",
            "clinical_service_disruption": "Prevents nursing staff from accessing web-based patient status boards and electronic medication delivery consoles.",
            "data_integrity_loss": "No data corruption, but complete unavailability of clinical user interface."
        },
        "mitigation_directive": {
            "action": "BLOCK_FLOW",
            "priority": 2,
            "sdn_action": "DROP",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 3600,
                "idle_timeout_sec": 180,
                "vlan_quarantine": False
            },
            "technical_rationale": "Install SDN rule dropping TCP sessions from offending source IP exhibiting lingering incomplete HTTP handshakes.",
            "clinical_rationale": "Frees exhausted clinical server thread pools and restores accessibility to nurse telemetry consoles."
        }
    },

    "rudeadyet": {
        "threat_summary": "R.U.D.Y. (R U Dead Yet?) slow POST data injection attack sending microscopic body chunks over extended intervals to exhaust clinical web server worker processes.",
        "threat_type": "Application Layer Denial of Service (Slow HTTP POST)",
        "severity": "HIGH",
        "confidence_score": 0.94,
        "mitre_attack": {
            "tactic": "Impact",
            "tactic_id": "TA0040",
            "technique_id": "T1499",
            "technique_name": "Endpoint Denial of Service",
            "sub_technique_id": "T1499.003",
            "sub_technique_name": "Application Exhaustion Flood",
            "description": "Adversary specifies large Content-Length headers and transmits single-byte payloads at long delays, tying up server worker threads."
        },
        "cwe_cve_references": [
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "HIGH",
            "clinical_service_disruption": "Locks up hospital clinical form submission gateways (e.g. prescription approval and lab order entry).",
            "data_integrity_loss": "Stalls medical order routing."
        },
        "mitigation_directive": {
            "action": "BLOCK_FLOW",
            "priority": 2,
            "sdn_action": "DROP",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 3600,
                "idle_timeout_sec": 120,
                "vlan_quarantine": False
            },
            "technical_rationale": "Issue SDN flow-mod command to terminate slow POST flows from attacker IP with idle timeouts shorter than typical payload interval.",
            "clinical_rationale": "Prevents clinical web forms from hanging, ensuring pharmacy orders and patient admission data remain responsive."
        }
    },

    "slowread": {
        "threat_summary": "Slow Read HTTP DoS maintaining tiny TCP window advertisements, forcing medical diagnostic servers to retain large response buffers in memory.",
        "threat_type": "Application Layer Denial of Service (Slow Read TCP Window)",
        "severity": "HIGH",
        "confidence_score": 0.92,
        "mitre_attack": {
            "tactic": "Impact",
            "tactic_id": "TA0040",
            "technique_id": "T1499",
            "technique_name": "Endpoint Denial of Service",
            "sub_technique_id": "T1499.003",
            "sub_technique_name": "Application Exhaustion Flood",
            "description": "Adversary advertises a near-zero TCP receive window, forcing server to throttle outbound data while holding full response in socket buffers."
        },
        "cwe_cve_references": [
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "MODERATE",
            "clinical_service_disruption": "Saturates PACS imaging server memory pools, slowing down emergency CT scan retrieval.",
            "data_integrity_loss": "No data corruption, but delayed clinical diagnostic turnarounds."
        },
        "mitigation_directive": {
            "action": "BLOCK_FLOW",
            "priority": 2,
            "sdn_action": "DROP",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 3600,
                "idle_timeout_sec": 120,
                "vlan_quarantine": False
            },
            "technical_rationale": "Deploy OpenFlow drop rule matching the attacker TCP 5-tuple exhibiting zero-window starvation patterns.",
            "clinical_rationale": "Recovers socket memory buffers on PACS diagnostic servers for acute radiology workflows."
        }
    },

    "netscan": {
        "threat_summary": "Systematic network reconnaissance scanning clinical subnets to discover active IoMT devices, open DICOM nodes, and exposed medical administration interfaces.",
        "threat_type": "Network Service Discovery / Port Reconnaissance",
        "severity": "MEDIUM",
        "confidence_score": 0.95,
        "mitre_attack": {
            "tactic": "Discovery",
            "tactic_id": "TA0007",
            "technique_id": "T1046",
            "technique_name": "Network Service Discovery",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "Adversary performs SYN or connect scans across medical IP ranges to map reachable clinical endpoints and services."
        },
        "cwe_cve_references": [
            {"id": "CWE-200", "description": "Exposure of Sensitive Information"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "LOW",
            "clinical_service_disruption": "Light telemetry jitter during scan; acts as the precursor to targeted exploit delivery against fragile medical appliances.",
            "data_integrity_loss": "None currently; high reconnaissance risk for downstream breach."
        },
        "mitigation_directive": {
            "action": "RATE_LIMIT",
            "priority": 3,
            "sdn_action": "METER_POLICING",
            "parameters": {
                "rate_limit_pps": 10,
                "hard_timeout_sec": 1800,
                "idle_timeout_sec": 300,
                "vlan_quarantine": False
            },
            "technical_rationale": "Clamp probe traffic from scanning IP to 10 pps via OpenFlow meter and log flagged targets for SOC review.",
            "clinical_rationale": "Obstructs adversary mapping of clinical devices while avoiding disruption to any legitimate medical device discovery protocols."
        }
    },

    "camoverflow": {
        "threat_summary": "MAC address flooding (CAM table overflow) saturating switch memory, forcing fallback into fail-open hub broadcast mode across the clinical network.",
        "threat_type": "Layer 2 Infrastructure Attack (CAM Table Overflow)",
        "severity": "CRITICAL",
        "confidence_score": 0.96,
        "mitre_attack": {
            "tactic": "Impact / Lateral Movement",
            "tactic_id": "TA0040",
            "technique_id": "T1498",
            "technique_name": "Network Denial of Service",
            "sub_technique_id": "T1498.001",
            "sub_technique_name": "Direct Network Flood",
            "description": "Adversary transmits floods of frames with randomized source MACs to fill the switch CAM table, causing frames to flood all ports."
        },
        "cwe_cve_references": [
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "CRITICAL",
            "clinical_service_disruption": "Hospital switch fails open, exposing confidential patient biometrics to promiscuous eavesdropping and causing network broadcast storms.",
            "data_integrity_loss": "Extreme risk of hospital-wide packet loss and unencrypted PHI exposure."
        },
        "mitigation_directive": {
            "action": "ISOLATE_PORT",
            "priority": 1,
            "sdn_action": "PORT_SECURITY_SHUTDOWN",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 7200,
                "idle_timeout_sec": 600,
                "vlan_quarantine": True
            },
            "technical_rationale": "SDN controller sends port-mod command disabling switch port originating the randomized MAC flood and enforcing port security limit.",
            "clinical_rationale": "Halts broadcast storm instantly, preserving medical telemetry delivery to ICU monitoring stations."
        }
    },

    # --------------------------------------------------------
    # MedSec-25 Vectors
    # --------------------------------------------------------
    "Initial access": {
        "threat_summary": "Exploitation of public-facing clinical service or unauthenticated medical device interface to gain preliminary network footprint.",
        "threat_type": "Initial Foothold / Perimeter Breach",
        "severity": "CRITICAL",
        "confidence_score": 0.95,
        "mitre_attack": {
            "tactic": "Initial Access",
            "tactic_id": "TA0001",
            "technique_id": "T1190",
            "technique_name": "Exploit Public-Facing Application",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "Adversary takes advantage of a weakness in an Internet-facing medical service to gain initial ingress."
        },
        "cwe_cve_references": [
            {"id": "CWE-287", "description": "Improper Authentication"},
            {"id": "CWE-119", "description": "Improper Restriction of Operations within the Bounds of a Memory Buffer"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "HIGH",
            "clinical_service_disruption": "Compromised edge device can be leveraged to pivot into life-critical telemetry segments.",
            "data_integrity_loss": "Risk of unauthorized administrative access to clinical systems."
        },
        "mitigation_directive": {
            "action": "BLOCK_FLOW",
            "priority": 1,
            "sdn_action": "DROP",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 7200,
                "idle_timeout_sec": 600,
                "vlan_quarantine": False
            },
            "technical_rationale": "Immediately drop all ingress packets matching the external attacker IP address at the SDN boundary gateway.",
            "clinical_rationale": "Prevents adversary from establishing persistent command and control into hospital infrastructure."
        }
    },

    "Reconnaissance": {
        "threat_summary": "Active scanning and vulnerability probing across medical network segments to identify unpatched infusion pumps, PACS servers, and clinical gateways.",
        "threat_type": "Active Pre-Attack Reconnaissance",
        "severity": "MEDIUM",
        "confidence_score": 0.94,
        "mitre_attack": {
            "tactic": "Reconnaissance",
            "tactic_id": "TA0043",
            "technique_id": "T1595",
            "technique_name": "Active Scanning",
            "sub_technique_id": "T1595.002",
            "sub_technique_name": "Vulnerability Scanning",
            "description": "Adversary probes clinical network hosts to identify vulnerable software versions and exploitable services."
        },
        "cwe_cve_references": [
            {"id": "CWE-200", "description": "Exposure of Sensitive Information"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "LOW",
            "clinical_service_disruption": "Minimal immediate disruption; high precursor risk of targeted medical device compromise.",
            "data_integrity_loss": "No data corruption currently observed."
        },
        "mitigation_directive": {
            "action": "RATE_LIMIT",
            "priority": 3,
            "sdn_action": "METER_POLICING",
            "parameters": {
                "rate_limit_pps": 15,
                "hard_timeout_sec": 1800,
                "idle_timeout_sec": 300,
                "vlan_quarantine": False
            },
            "technical_rationale": "Enforce SDN flow meter restricting scan probe packet rates to 15 pps and redirect suspect flows to honeynet.",
            "clinical_rationale": "Limits adversary information gathering without risking false-positive interruption of legitimate medical sweeps."
        }
    },

    "Lateral movement": {
        "threat_summary": "Adversary pivoting across internal hospital segments from a non-clinical workstation towards sensitive IoMT ICU and surgical subnets.",
        "threat_type": "Internal Lateral Movement",
        "severity": "CRITICAL",
        "confidence_score": 0.96,
        "mitre_attack": {
            "tactic": "Lateral Movement",
            "tactic_id": "TA0008",
            "technique_id": "T1021",
            "technique_name": "Remote Services",
            "sub_technique_id": "T1021.002",
            "sub_technique_name": "SMB/Windows Admin Shares",
            "description": "Adversary logs into remote systems using credentials or exploits internal services to traverse into medical segments."
        },
        "cwe_cve_references": [
            {"id": "CWE-284", "description": "Improper Access Control"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "CRITICAL",
            "clinical_service_disruption": "Compromise of surgical and ICU VLANs directly threatens patient monitoring and medical device actuation.",
            "data_integrity_loss": "High risk of malicious payloads deployed directly onto clinical hardware."
        },
        "mitigation_directive": {
            "action": "QUARANTINE_HOST",
            "priority": 1,
            "sdn_action": "SET_VLAN_QUARANTINE",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 14400,
                "idle_timeout_sec": 1200,
                "vlan_quarantine_id": 999
            },
            "technical_rationale": "Trigger immediate micro-segmentation flow isolation on SDN controller to sever east-west inter-VLAN forwarding for the compromised node.",
            "clinical_rationale": "Contains lateral spread before attacker can reach life-support equipment or EHR databases."
        }
    },

    "Exfiltration": {
        "threat_summary": "Unauthorized transfer of sensitive patient health records (PHI), diagnostic telemetry, and clinical records out of the hospital perimeter.",
        "threat_type": "Data Exfiltration Over Network Protocol",
        "severity": "CRITICAL",
        "confidence_score": 0.97,
        "mitre_attack": {
            "tactic": "Exfiltration",
            "tactic_id": "TA0010",
            "technique_id": "T1041",
            "technique_name": "Exfiltration Over C2 Channel",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "Adversary steals clinical telemetry or PHI data by transferring it through established command and control channels."
        },
        "cwe_cve_references": [
            {"id": "CWE-200", "description": "Exposure of Sensitive Information"},
            {"id": "CWE-319", "description": "Cleartext Transmission of Sensitive Information"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "HIGH",
            "clinical_service_disruption": "HIPAA compliance catastrophe; potential weaponization of stolen patient biometric telemetry.",
            "data_integrity_loss": "Severe loss of patient confidentiality and medical record privacy."
        },
        "mitigation_directive": {
            "action": "BLOCK_FLOW",
            "priority": 1,
            "sdn_action": "DROP",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 7200,
                "idle_timeout_sec": 600,
                "vlan_quarantine": False
            },
            "technical_rationale": "Install immediate egress blocking rule on perimeter SDN switch dropping outbound TCP connection and terminating active flow socket.",
            "clinical_rationale": "Stops exfiltration of patient health records instantly to prevent data theft."
        }
    },

    # --------------------------------------------------------
    # Non-Attack / Normal Clinical Baseline Traffic
    # --------------------------------------------------------
    "normal": {
        "threat_summary": "Authorized, routine IoMT clinical device communication exhibiting nominal flow rates, valid protocol handshakes, and expected clinical baseline behavior.",
        "threat_type": "Benign / Nominal Clinical Traffic",
        "severity": "INFORMATIONAL",
        "confidence_score": 0.98,
        "mitre_attack": {
            "tactic": "None (Benign)",
            "tactic_id": "N/A",
            "technique_id": "None",
            "technique_name": "Nominal IoMT Device Operation",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "Flow exhibits standard baseline medical communication with no malicious or anomalous indicators."
        },
        "cwe_cve_references": [],
        "clinical_impact_assessment": {
            "patient_safety_risk": "NEGLIGIBLE",
            "clinical_service_disruption": "None. Continuous, uninterrupted patient physiological monitoring active.",
            "data_integrity_loss": "None. Authenticated, uncorrupted clinical medical telemetry."
        },
        "mitigation_directive": {
            "action": "ALLOW_FLOW",
            "priority": 5,
            "sdn_action": "FORWARD",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 0,
                "idle_timeout_sec": 0,
                "vlan_quarantine": False
            },
            "technical_rationale": "Permit legitimate medical device flow to pass through OpenFlow switches unimpeded without filtering or rate limiting.",
            "clinical_rationale": "Maintains continuous, real-time patient physiological monitoring and uninterrupted medical device operation."
        }
    },

    "Benign": {
        "threat_summary": "Authorized, routine IoMT clinical device communication exhibiting nominal flow rates, valid protocol handshakes, and expected clinical baseline behavior.",
        "threat_type": "Benign / Nominal Clinical Traffic",
        "severity": "INFORMATIONAL",
        "confidence_score": 0.98,
        "mitre_attack": {
            "tactic": "None (Benign)",
            "tactic_id": "N/A",
            "technique_id": "None",
            "technique_name": "Nominal IoMT Device Operation",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "Flow exhibits standard baseline medical communication with no malicious or anomalous indicators."
        },
        "cwe_cve_references": [],
        "clinical_impact_assessment": {
            "patient_safety_risk": "NEGLIGIBLE",
            "clinical_service_disruption": "None. Continuous, uninterrupted patient physiological monitoring active.",
            "data_integrity_loss": "None. Authenticated, uncorrupted clinical medical telemetry."
        },
        "mitigation_directive": {
            "action": "ALLOW_FLOW",
            "priority": 5,
            "sdn_action": "FORWARD",
            "parameters": {
                "rate_limit_pps": 0,
                "hard_timeout_sec": 0,
                "idle_timeout_sec": 0,
                "vlan_quarantine": False
            },
            "technical_rationale": "Permit legitimate medical device flow to pass through OpenFlow switches unimpeded without filtering or rate limiting.",
            "clinical_rationale": "Maintains continuous, real-time patient physiological monitoring and uninterrupted medical device operation."
        }
    },

    # --------------------------------------------------------
    # Generic entries used by the rule-based offline engine
    # --------------------------------------------------------
    "http_dos": {
        "threat_summary": "Anomalous HTTP flow towards a clinical web service consistent with an application-layer denial-of-service pattern (e.g. slow headers/body, slow read or range-header amplification).",
        "threat_type": "Application Layer Denial of Service (HTTP)",
        "severity": "HIGH",
        "confidence_score": 0.60,
        "mitre_attack": {
            "tactic": "Impact",
            "tactic_id": "TA0040",
            "technique_id": "T1499",
            "technique_name": "Endpoint Denial of Service",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "Flow to an HTTP service flagged by SAFE; port-based rule maps it to endpoint DoS at technique level."
        },
        "cwe_cve_references": [
            {"id": "CWE-400", "description": "Uncontrolled Resource Consumption"}
        ],
        "clinical_impact_assessment": {
            "patient_safety_risk": "HIGH",
            "clinical_service_disruption": "Possible exhaustion of clinical web gateway resources.",
            "data_integrity_loss": "Availability impact; no direct data corruption indicated."
        },
        "mitigation_directive": {
            "action": "RATE_LIMIT",
            "priority": 2,
            "sdn_action": "METER_POLICING",
            "parameters": {"rate_limit_pps": 50, "hard_timeout_sec": 1800, "idle_timeout_sec": 300, "vlan_quarantine": False},
            "technical_rationale": "Rate-limit the source towards the HTTP service instead of a hard drop, because the attribution comes from a port rule.",
            "clinical_rationale": "Limits resource exhaustion while keeping legitimate clinical web access possible."
        }
    },

    "unclassified_anomaly": {
        "threat_summary": "Flow flagged as anomalous by SAFE that matches no rule of the offline engine; attack type undetermined.",
        "threat_type": "Unclassified Network Anomaly",
        "severity": "MEDIUM",
        "confidence_score": 0.40,
        "mitre_attack": {
            "tactic": "Unknown",
            "tactic_id": "N/A",
            "technique_id": "None",
            "technique_name": "Unattributed anomaly",
            "sub_technique_id": None,
            "sub_technique_name": None,
            "description": "No ATT&CK technique can be attributed from the available evidence by the rule-based engine."
        },
        "cwe_cve_references": [],
        "clinical_impact_assessment": {
            "patient_safety_risk": "MODERATE",
            "clinical_service_disruption": "Unknown; analyst review required.",
            "data_integrity_loss": "Unknown."
        },
        "mitigation_directive": {
            "action": "RATE_LIMIT",
            "priority": 3,
            "sdn_action": "METER_POLICING",
            "parameters": {"rate_limit_pps": 100, "hard_timeout_sec": 900, "idle_timeout_sec": 300, "vlan_quarantine": False},
            "technical_rationale": "Conservative rate limit pending SOC analyst review.",
            "clinical_rationale": "Avoids fully blocking a possibly legitimate clinical flow on uncertain evidence."
        }
    }
}


# ============================================================
# EVALUATION HELPERS
# ============================================================

def _technique_root(tid: Optional[str]) -> Optional[str]:
    if not tid or str(tid) in ("None", "N/A", "null"):
        return None
    return str(tid).split(".")[0]


def expected_technique(label: Optional[str]) -> Optional[str]:
    """Ground-truth attack label -> expected technique (from the knowledge base)."""
    if label is None:
        return None
    entry = IOMT_ATTACK_KNOWLEDGE_BASE.get(label) or IOMT_ATTACK_KNOWLEDGE_BASE.get(str(label).lower())
    if not entry:
        return None
    return _technique_root(entry["mitre_attack"]["technique_id"])


def compute_reasoning_metrics(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    gts = [a.get("ground_truth") or {} for a in analyses]
    actions = [a.get("mitigation_directive", {}).get("action") for a in analyses]

    benign = [i for i, g in enumerate(gts) if g.get("is_attack") is False]
    attacks = [i for i, g in enumerate(gts) if g.get("is_attack") is True]
    flagged_attacks = [i for i in attacks if analyses[i].get("safe_verdict") == "Anomaly"]

    scored = [i for i in flagged_attacks if expected_technique(gts[i].get("label"))]
    correct = sum(
        1 for i in scored
        if _technique_root(analyses[i].get("mitre_attack", {}).get("technique_id"))
        == expected_technique(gts[i].get("label")))

    benign_restricted = sum(1 for i in benign if actions[i] != "ALLOW_FLOW")
    attacks_allowed = sum(1 for i in attacks if actions[i] == "ALLOW_FLOW")
    attacks_actioned = len(attacks) - attacks_allowed

    meta = [a.get("reasoning_metadata", {}) for a in analyses]
    llm_runs = [m for m in meta if m.get("llm_output_valid") is not None]
    latencies = [m.get("latency_ms") for m in meta if m.get("latency_ms") is not None]

    def pct(n, d):
        return round(100.0 * n / d, 2) if d else None

    return {
        "flows_analysed": len(analyses),
        "ground_truth_attacks": len(attacks),
        "ground_truth_benign": len(benign),
        "attacks_flagged_by_safe": len(flagged_attacks),
        "attribution_scored_flows": len(scored),
        "attribution_accuracy_percent": pct(correct, len(scored)),
        "attack_mitigation_coverage_percent": pct(attacks_actioned, len(attacks)),
        "missed_attack_rate_percent": pct(attacks_allowed, len(attacks)),
        "false_mitigation_rate_percent": pct(benign_restricted, len(benign)),
        "benign_flows_restricted": benign_restricted,
        "llm_valid_output_percent": pct(sum(1 for m in llm_runs if m.get("llm_output_valid")), len(llm_runs)),
        "fallback_used_count": sum(1 for m in meta if m.get("fallback_used")),
        "mean_reasoning_latency_ms": round(float(np.mean(latencies)), 3) if latencies else None,
        "notes": "Attribution accuracy is technique-level (sub-techniques ignored) on ground-truth attacks "
                 "that SAFE flagged and whose label has a technique in the knowledge base.",
    }


# ============================================================
# FOUNDATION-SEC REASONER
# ============================================================

class FoundationSecReasoner:

    def __init__(self, dataset_name: str, backend: str = "offline",
                 model_id: str = "fdtn-ai/Foundation-Sec-8B-Instruct",
                 api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.dataset_name = dataset_name
        self.backend = backend.lower()
        self.model_id = model_id
        self.api_url = api_url or os.environ.get("FOUNDATION_SEC_API_URL")
        self.api_key = api_key or os.environ.get("FOUNDATION_SEC_API_KEY")
        self.results_dir = os.path.join(REASONING_RESULTS_DIR, dataset_name)
        os.makedirs(self.results_dir, exist_ok=True)

        self._tokenizer = None
        self._model = None
        if self.backend == "huggingface":
            self._init_huggingface()
        if self.backend == "api" and not self.api_url:
            print("[M5] WARNING: backend 'api' but FOUNDATION_SEC_API_URL is not set; using rule-based engine.")
            self.backend = "offline"

    # --------------------------------------------------------
    # Backends
    # --------------------------------------------------------

    def _init_huggingface(self):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            kwargs = {}
            if torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                if os.environ.get("FOUNDATION_SEC_4BIT") == "1":
                    from transformers import BitsAndBytesConfig
                    kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                else:
                    kwargs["torch_dtype"] = torch.bfloat16
            else:
                print("[M5] WARNING: no GPU found. An 8B model on CPU needs ~32 GB RAM and is very slow.")
                kwargs["torch_dtype"] = torch.float32
            print(f"[M5] Loading {self.model_id} ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self._model.eval()
            print("[M5] Model loaded.")
        except Exception as e:
            print(f"[M5] WARNING: could not load the model ({e}). Using the rule-based engine.")
            self.backend = "offline"

    @staticmethod
    def sanitize_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
        """Remove everything that reveals the answer before it reaches a model."""
        clean = {k: v for k, v in item.items() if k not in ("ground_truth", "is_attack", "raw_traffic_label")}
        return clean

    def build_messages(self, item: Dict[str, Any]) -> List[Dict[str, str]]:
        evidence = json.dumps(self.sanitize_evidence(item), indent=2, default=str)
        user = ("Analyse this IoMT network flow evidence:\n\n```json\n" + evidence + "\n```\n\n"
                "Respond with one JSON object following this schema:\n" + OUTPUT_SCHEMA)
        return [{"role": "system", "content": FOUNDATION_SEC_SYSTEM_PROMPT},
                {"role": "user", "content": user}]

    def _generate_hf(self, messages: List[Dict[str, str]]) -> str:
        import torch
        inputs = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(inputs, max_new_tokens=700, do_sample=False,
                                       pad_token_id=self._tokenizer.eos_token_id)
        return self._tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    def _generate_api(self, messages: List[Dict[str, str]]) -> str:
        import urllib.request
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps({"model": self.model_id, "messages": messages,
                              "temperature": 0, "max_tokens": 700}).encode("utf-8")
        req = urllib.request.Request(self.api_url, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=float(os.environ.get("FOUNDATION_SEC_TIMEOUT", "900"))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    # --------------------------------------------------------
    # Output validation
    # --------------------------------------------------------

    @staticmethod
    def parse_and_validate(text: str) -> Optional[Dict[str, Any]]:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
        try:
            mit = obj["mitigation_directive"]
            att = obj["mitre_attack"]
            mit["action"] = str(mit["action"]).upper()
            obj["severity"] = str(obj["severity"]).upper()
            if mit["action"] not in ALLOWED_ACTIONS or obj["severity"] not in ALLOWED_SEVERITY:
                return None
            tid = str(att.get("technique_id", "None"))
            if tid not in ("None", "N/A") and not TECHNIQUE_RE.match(tid):
                return None
            obj["confidence_score"] = float(min(1.0, max(0.0, float(obj.get("confidence_score", 0.5)))))
            obj.setdefault("cwe_cve_references", [])
            obj.setdefault("clinical_impact_assessment", {})
            obj.setdefault("threat_summary", "")
            obj.setdefault("threat_type", "")
            return obj
        except (KeyError, TypeError, ValueError):
            return None

    # --------------------------------------------------------
    # Rule-based engine (offline)
    # --------------------------------------------------------

    def rule_based_key(self, item: Dict[str, Any]) -> str:
        if item.get("detection", {}).get("verdict") != "Anomaly":
            return "normal"
        flow = item.get("network_flow", {}) or {}
        port = flow.get("destination_port")
        proto = str(flow.get("protocol") or "").upper()
        syn, ack = flow.get("syn_flags"), flow.get("ack_flags")
        bwd = flow.get("bwd_packets")
        if port in (1883, 8883):
            return "mqttmalaria"
        if proto == "ARP":
            return "arpspoofing"
        if syn and not ack and (bwd is not None and bwd <= 1):
            return "netscan"
        if port in (80, 8080, 443):
            return "http_dos"
        return "unclassified_anomaly"

    def _rule_based(self, item: Dict[str, Any]) -> Dict[str, Any]:
        k = IOMT_ATTACK_KNOWLEDGE_BASE[self.rule_based_key(item)]
        return {key: json.loads(json.dumps(k[key])) for key in
                ["threat_summary", "threat_type", "severity", "confidence_score", "mitre_attack",
                 "cwe_cve_references", "clinical_impact_assessment", "mitigation_directive"]}

    # --------------------------------------------------------
    # One evidence item
    # --------------------------------------------------------

    def analyse_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        llm_valid, fallback, raw_output = None, False, None
        engine = "rule-based (offline)"

        if self.backend in ("huggingface", "api"):
            engine = f"{self.model_id} ({self.backend})"
            try:
                messages = self.build_messages(item)
                raw_output = self._generate_hf(messages) if self.backend == "huggingface" else self._generate_api(messages)
                analysis = self.parse_and_validate(raw_output)
            except Exception as e:
                print(f"[M5] model call failed for {item.get('evidence_id')}: {e}")
                analysis = None
            llm_valid = analysis is not None
            if analysis is None:
                fallback = True
                analysis = self._rule_based(item)
        else:
            analysis = self._rule_based(item)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        flow = item.get("network_flow", {}) or {}
        det = item.get("detection", {}) or {}
        analysis["mitigation_directive"]["target"] = {
            "source_ip": flow.get("source_ip"),
            "destination_ip": flow.get("destination_ip"),
            "source_port": flow.get("source_port"),
            "destination_port": flow.get("destination_port"),
            "protocol": flow.get("protocol"),
        }
        return {
            "evidence_id": item.get("evidence_id"),
            "flow_id": item.get("flow_id"),
            "dataset": self.dataset_name,
            "safe_verdict": det.get("verdict"),
            # kept for compatibility: True = flagged by SAFE (prediction, NOT ground truth)
            "is_attack": det.get("verdict") == "Anomaly",
            **analysis,
            "reasoning_metadata": {
                "engine": engine,
                "backend": self.backend,
                "llm_output_valid": llm_valid,
                "fallback_used": fallback,
                "latency_ms": round(latency_ms, 3),
                "anomaly_score": det.get("anomaly_score"),
                "device_role_assumed": (item.get("device_context") or {}).get("role"),
                "reasoned_at": datetime.now(timezone.utc).isoformat(),
                "raw_model_output": raw_output if (raw_output and not llm_valid) else None,
            },
            # attached AFTER reasoning, for evaluation only
            "ground_truth": item.get("ground_truth"),
        }

    def analyze_evidence(self, evidence_file: str) -> List[Dict[str, Any]]:
        if not os.path.exists(evidence_file):
            raise FileNotFoundError(f"Evidence file not found: {evidence_file}")
        with open(evidence_file, "r", encoding="utf-8") as f:
            items = json.load(f).get("evidence_items", [])
        print(f"\n[M5] Reasoning over {len(items)} evidence items (backend: {self.backend})...")
        analyses = []
        for i, item in enumerate(items):
            a = self.analyse_item(item)
            analyses.append(a)
            gt = (a.get("ground_truth") or {}).get("label", "?")
            print(f"  [{i + 1:03d}/{len(items):03d}] {a['evidence_id']} SAFE={a['safe_verdict']:<7} "
                  f"truth={str(gt):<16} -> {a['mitre_attack'].get('technique_id', 'None'):<9} "
                  f"{a['severity']:<13} {a['mitigation_directive']['action']}")
        return analyses

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save_results(self, analyses: List[Dict[str, Any]]) -> Dict[str, str]:
        reasoning_path = os.path.join(self.results_dir, "foundation_sec_reasoning.json")
        with open(reasoning_path, "w", encoding="utf-8") as f:
            json.dump({"reasoning_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": self.dataset_name,
                "model_identifier": self.model_id if self.backend != "offline" else None,
                "reasoning_backend": self.backend,
                "engine": "rule-based (offline)" if self.backend == "offline" else self.model_id,
                "total_analyzed": len(analyses)},
                "analyses": analyses}, f, indent=4, default=str)

        tactic_counts, technique_counts, severity_counts, action_counts = {}, {}, {}, {}
        for item in analyses:
            att = item.get("mitre_attack", {})
            tech = f"{att.get('technique_id', 'None')} - {att.get('technique_name', 'Unknown')}"
            for d, key in ((tactic_counts, att.get("tactic", "Unknown")), (technique_counts, tech),
                           (severity_counts, item.get("severity", "MEDIUM")),
                           (action_counts, item.get("mitigation_directive", {}).get("action", "UNKNOWN"))):
                d[key] = d.get(key, 0) + 1

        summary = {
            "dataset": self.dataset_name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "reasoning_backend": self.backend,
            "total_incidents": len(analyses),
            "severity_distribution": severity_counts,
            "mitre_tactics_distribution": tactic_counts,
            "mitre_techniques_distribution": technique_counts,
            "sdn_mitigation_actions": action_counts,
            "evaluation": compute_reasoning_metrics(analyses),
        }
        summary_path = os.path.join(self.results_dir, "mitre_attack_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        report_path = os.path.join(self.results_dir, "threat_report.md")
        self._generate_markdown_report(report_path, summary, analyses)
        print(f"\n[M5] Saved:\n  {reasoning_path}\n  {summary_path}\n  {report_path}")
        return {"reasoning": reasoning_path, "summary": summary_path, "report": report_path}

    def _generate_markdown_report(self, report_path, summary, analyses):
        ev = summary["evaluation"]
        engine = "Rule-based offline engine (not a language model)" if self.backend == "offline" \
            else f"{self.model_id} via {self.backend}"
        fmt = lambda v: "N/A" if v is None else f"{v}%"
        lines = [
            "# IoMT SOC Threat Intelligence Report",
            f"**Dataset:** `{self.dataset_name}`  ",
            f"**Reasoning engine:** {engine}  ",
            f"**Generated:** `{summary['generated_at']}`  ",
            f"**Flows analysed:** {summary['total_incidents']} (sampled from SAFE test predictions)", "",
            "## 1. Evaluation against ground truth", "",
            "| Metric | Value |", "| :--- | :--- |",
            f"| Attribution accuracy (technique level, {ev['attribution_scored_flows']} flagged attacks) | {fmt(ev['attribution_accuracy_percent'])} |",
            f"| Attack mitigation coverage | {fmt(ev['attack_mitigation_coverage_percent'])} |",
            f"| Missed-attack rate (attack given ALLOW_FLOW) | {fmt(ev['missed_attack_rate_percent'])} |",
            f"| False-mitigation rate (benign given restrictive action) | {fmt(ev['false_mitigation_rate_percent'])} |",
            f"| Valid model output | {fmt(ev['llm_valid_output_percent'])} |",
            f"| Mean reasoning latency | {ev['mean_reasoning_latency_ms']} ms |", "",
            "## 2. Severity breakdown", "", "| Severity | Count |", "| :--- | :--- |",
        ]
        lines += [f"| {k} | {v} |" for k, v in summary["severity_distribution"].items()]
        lines += ["", "## 3. Recommended SDN actions", "", "| Action | Count |", "| :--- | :--- |"]
        lines += [f"| `{k}` | {v} |" for k, v in summary["sdn_mitigation_actions"].items()]
        lines += ["", "## 4. MITRE ATT&CK techniques", "", "| Technique | Count |", "| :--- | :--- |"]
        lines += [f"| {k} | {v} |" for k, v in summary["mitre_techniques_distribution"].items()]
        lines += ["", "## 5. Sample incidents", ""]
        for item in [a for a in analyses if a.get("safe_verdict") == "Anomaly"][:5]:
            att, mit = item.get("mitre_attack", {}), item.get("mitigation_directive", {})
            t = mit.get("target", {})
            lines += [
                f"### {item['evidence_id']}: {item.get('threat_type')}",
                f"- **Ground truth:** {(item.get('ground_truth') or {}).get('label')}",
                f"- **Flow:** `{t.get('source_ip')}:{t.get('source_port')} -> {t.get('destination_ip')}:{t.get('destination_port')}` ({t.get('protocol')})",
                f"- **ATT&CK:** `{att.get('technique_id')}` {att.get('technique_name')} (tactic: {att.get('tactic')})",
                f"- **Severity:** {item.get('severity')} (confidence {item.get('confidence_score')})",
                f"- **Action:** `{mit.get('action')}` - {mit.get('technical_rationale')}", ""]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def process_dataset_reasoning(dataset_name: str, backend: str = "offline",
                              model_id: str = "fdtn-ai/Foundation-Sec-8B-Instruct", max_samples: int = 100):
    evidence_file = os.path.join(REASONING_RESULTS_DIR, dataset_name, "threat_evidence.json")
    if not os.path.exists(evidence_file):
        from .threat_evidence import ThreatEvidenceBuilder
        evidence_file = ThreatEvidenceBuilder(dataset_name).build_and_save_evidence(max_samples=max_samples)
    reasoner = FoundationSecReasoner(dataset_name=dataset_name, backend=backend, model_id=model_id)
    return reasoner.save_results(reasoner.analyze_evidence(evidence_file))


def main():
    parser = argparse.ArgumentParser(description="M5 - Foundation-Sec Threat Reasoning")
    parser.add_argument("--dataset", choices=["IoMT-TrafficData", "MedSec-25", "all"], default="all")
    parser.add_argument("--backend", choices=["offline", "huggingface", "api"], default="offline")
    parser.add_argument("--model-id", default="fdtn-ai/Foundation-Sec-8B-Instruct")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    datasets = ["IoMT-TrafficData", "MedSec-25"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        process_dataset_reasoning(ds, args.backend, args.model_id, args.samples)


if __name__ == "__main__":
    main()