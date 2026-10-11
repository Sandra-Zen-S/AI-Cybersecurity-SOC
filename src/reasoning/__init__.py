"""
AI Cybersecurity SOC Assistant - Reasoning Layer
================================================
Modules:
- M4: Threat Evidence Builder (threat_evidence.py)
- M5: Foundation-Sec Threat Analysis (foundation_sec.py)
"""

from .threat_evidence import ThreatEvidenceBuilder
from .foundation_sec import FoundationSecReasoner

__all__ = [
    "ThreatEvidenceBuilder",
    "FoundationSecReasoner"
]
