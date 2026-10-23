# IoMT SOC Threat Intelligence Report
**Dataset:** `MedSec-25`  
**Reasoning engine:** foundation-sec via api  
**Generated:** `2026-10-18T12:57:07.240275+00:00`  
**Flows analysed:** 100 (sampled from SAFE test predictions)

## 1. Evaluation against ground truth

| Metric | Value |
| :--- | :--- |
| Attribution accuracy (technique level, 50 flagged attacks) | 4.0% |
| Attack mitigation coverage | 53.61% |
| Missed-attack rate (attack given ALLOW_FLOW) | 46.39% |
| False-mitigation rate (benign given restrictive action) | 0.0% |
| Valid model output | 100.0% |
| Mean reasoning latency | 18311.735 ms |

## 2. Severity breakdown

| Severity | Count |
| :--- | :--- |
| INFORMATIONAL | 30 |
| LOW | 47 |
| MEDIUM | 21 |
| HIGH | 2 |

## 3. Recommended SDN actions

| Action | Count |
| :--- | :--- |
| `ALLOW_FLOW` | 48 |
| `RATE_LIMIT` | 51 |
| `BLOCK_FLOW` | 1 |

## 4. MITRE ATT&CK techniques

| Technique | Count |
| :--- | :--- |
| None - None | 48 |
| T1046 - Network Service Scanning | 30 |
| T1083 - System Network Configuration Discovery | 1 |
| T1593 - Scan Network | 2 |
| T1041 - Exfiltration Over C2 Channel | 4 |
| T1059 - Command and Scripting Interpreter | 3 |
| T1048 - Exfiltration Over Alternative Protocol | 1 |
| T1071 - Application Layer Protocol | 6 |
| T1595 - Network Service Scanning | 2 |
| T1095 - Non-Application Layer Protocol | 1 |
| T1046 - Network Service Discovery | 2 |

## 5. Sample incidents

### EVID-MEDS-0005: Malicious
- **Ground truth:** Reconnaissance
- **Flow:** `192.168.1.108:54451 -> 192.168.1.101:44781` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** LOW (confidence 0.7)
- **Action:** `RATE_LIMIT` - Rate limiting the flow to prevent potential reconnaissance activities without completely blocking the device's functionality.

### EVID-MEDS-0007: Malicious
- **Ground truth:** Reconnaissance
- **Flow:** `192.168.1.108:54451 -> 192.168.1.1:32712` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** LOW (confidence 0.75)
- **Action:** `RATE_LIMIT` - Rate limiting will prevent excessive network usage without blocking potentially benign traffic.

### EVID-MEDS-0008: Malicious
- **Ground truth:** Lateral movement
- **Flow:** `192.168.1.100:0 -> 224.0.0.251:0` (0)
- **ATT&CK:** `T1083` System Network Configuration Discovery (tactic: Discovery)
- **Severity:** LOW (confidence 0.75)
- **Action:** `RATE_LIMIT` - Rate limiting the flow to 10 packets per second will prevent excessive probing without blocking potentially benign traffic.

### EVID-MEDS-0009: Malicious
- **Ground truth:** Reconnaissance
- **Flow:** `192.168.1.108:54451 -> 192.168.1.102:819` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** LOW (confidence 0.7)
- **Action:** `RATE_LIMIT` - Rate limiting will prevent excessive network usage without blocking potentially benign traffic.

### EVID-MEDS-0012: Malicious
- **Ground truth:** Reconnaissance
- **Flow:** `192.168.1.108:58094 -> 192.168.1.1:58427` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** LOW (confidence 0.7)
- **Action:** `RATE_LIMIT` - Rate limiting will prevent excessive network usage without blocking potentially benign traffic.
