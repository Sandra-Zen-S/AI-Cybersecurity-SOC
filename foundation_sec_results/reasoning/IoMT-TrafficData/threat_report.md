# IoMT SOC Threat Intelligence Report
**Dataset:** `IoMT-TrafficData`  
**Reasoning engine:** foundation-sec via api  
**Generated:** `2026-10-18T12:22:50.135389+00:00`  
**Flows analysed:** 100 (sampled from SAFE test predictions)

## 1. Evaluation against ground truth

| Metric | Value |
| :--- | :--- |
| Attribution accuracy (technique level, 50 flagged attacks) | 10.0% |
| Attack mitigation coverage | 100.0% |
| Missed-attack rate (attack given ALLOW_FLOW) | 0.0% |
| False-mitigation rate (benign given restrictive action) | 16.0% |
| Valid model output | 100.0% |
| Mean reasoning latency | 17904.446 ms |

## 2. Severity breakdown

| Severity | Count |
| :--- | :--- |
| INFORMATIONAL | 39 |
| LOW | 31 |
| MEDIUM | 30 |

## 3. Recommended SDN actions

| Action | Count |
| :--- | :--- |
| `ALLOW_FLOW` | 42 |
| `RATE_LIMIT` | 58 |

## 4. MITRE ATT&CK techniques

| Technique | Count |
| :--- | :--- |
| T1595 - Network Service Scanning | 31 |
| None - None | 37 |
| T1046 - Network Service Scanning | 25 |
| T1046 - Network Service Discovery | 2 |
| T1041 - Exfiltration Over C2 Channel | 1 |
| T1083 - File and Directory Discovery | 3 |
| T1048 - Exfiltration Over Alternative Protocol | 1 |

## 5. Sample incidents

### EVID-IOMT-0005: Malicious
- **Ground truth:** netscan
- **Flow:** `10.10.10.252:34761 -> 10.10.10.249:45459` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** MEDIUM (confidence 0.75)
- **Action:** `RATE_LIMIT` - Rate limiting the flow to prevent further reconnaissance activities.

### EVID-IOMT-0007: Malicious
- **Ground truth:** camoverflow
- **Flow:** `111.150.96.79:15083 -> 255.125.111.22:38625` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** MEDIUM (confidence 0.7)
- **Action:** `RATE_LIMIT` - Rate limiting the flow to 10 packets per second with a 300-second hard timeout and 60-second idle timeout can help mitigate potential reconnaissance activity without blocking benign traffic.

### EVID-IOMT-0008: Malicious
- **Ground truth:** netscan
- **Flow:** `10.10.10.252:41180 -> 10.10.10.249:24272` (TCP)
- **ATT&CK:** `T1595` Network Service Scanning (tactic: Reconnaissance)
- **Severity:** LOW (confidence 0.7)
- **Action:** `RATE_LIMIT` - Limit the rate of the flow to prevent further reconnaissance activities without completely blocking the device.

### EVID-IOMT-0009: Malicious
- **Ground truth:** netscan
- **Flow:** `10.10.10.252:63895 -> 10.10.10.249:10252` (TCP)
- **ATT&CK:** `T1595` Network Service Scanning (tactic: Reconnaissance)
- **Severity:** LOW (confidence 0.65)
- **Action:** `RATE_LIMIT` - Limit the rate of the flow to prevent potential reconnaissance activities from overwhelming the network.

### EVID-IOMT-0012: Malicious
- **Ground truth:** camoverflow
- **Flow:** `206.67.25.113:9222 -> 155.167.123.28:52111` (TCP)
- **ATT&CK:** `T1046` Network Service Scanning (tactic: Discovery)
- **Severity:** MEDIUM (confidence 0.8)
- **Action:** `RATE_LIMIT` - Rate limiting the flow to prevent excessive probing and potential data exfiltration.
