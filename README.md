# SmartIDS: Defensive Flow-Based Network Intrusion Detection System

An enterprise-ready, defensive network intrusion detection system designed to solve real-world ML deployment challenges without relying on artificial web-form entry or toy metrics.

---

## Architecture Overview

```
                      Raw Network Traffic (.pcap) / Ingested Flow CSV
                                           │
                                           ▼
                   ┌──────────────────────────────────────────────┐
                   │    Scapy Flow Extractor / Schema Contract    │
                   │  - Frozen 22-feature specification (v1.0.0)  │
                   │  - Hard-fail validation on schema mismatch   │
                   └──────────────────────┬───────────────────────┘
                                          │
                    ┌─────────────────────┴──────────────────────┐
                    ▼                                            ▼
      ┌───────────────────────────┐                ┌───────────────────────────┐
      │   Supervised Classifier   │                │   PyTorch Autoencoder     │
      │  - Leak-free sklearn pipe │                │  - Trained on BENIGN only │
      │  - LightGBM tuned w/ Optuna│               │  - 4-dim bottleneck       │
      │  - Threshold @ 0.1% FPR   │                │  - Reconstruction MSE     │
      └─────────────┬─────────────┘                └─────────────┬─────────────┘
                    │ p_supervised                               │ s_anomaly
                    └─────────────────────┬──────────────────────┘
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │          Hybrid Score Fusion Layer           │
                   │  - Critical: Confirmed Attack (p >= tau, s)  │
                   │  - High: Known Signature (p >= tau, !s)      │
                   │  - Medium: Zero-Day Novel Anomaly (!p, s)    │
                   │  - Low: Benign Baseline (!p, !s)             │
                   └──────────────────────┬───────────────────────┘
                                          │
                                          ▼
                   ┌──────────────────────────────────────────────┐
                   │       Alert Aggregator (Anti-Fatigue)        │
                   │  - Groups by (src_ip, port_pattern, window)  │
                   │  - Port scan burst -> 1 alert w/ hit_count   │
                   └──────────────────────┬───────────────────────┘
                                          │
                      ┌───────────────────┴───────────────────┐
                      ▼                                       ▼
       ┌─────────────────────────────┐         ┌─────────────────────────────┐
       │   FastAPI Ingestion & SSE   │         │    React + Vite Dashboard   │
       │  - Real-time event stream   │ ◄─────► │  - Live alerts table        │
       │  - Lazy on-demand SHAP (<15ms)│       │  - Operating point slider   │
       │  - KS-test drift monitor    │         │  - Drift flag & Gatekeeper  │
       └─────────────────────────────┘         └─────────────────────────────┘
```

---

## The Four Hard Defensive ML Problems Solved

### 1. Extreme Class Imbalance
* **Problem**: In real enterprise traffic, malicious flows account for $<3\%$ of volume. Relying on accuracy gives a false sense of security (a dummy model predicting 100% benign achieves $>97\%$ accuracy while missing all attacks).
* **Defensive Solution**:
  * Optimized **PR-AUC** (Precision-Recall Area Under Curve) via Optuna across 5-fold cross-validation.
  * Computed class weights strictly inside training folds.
  * **Headline Operating Metric**: Headline evaluation is **Recall @ 0.1% False Positive Rate (FPR)** ($1.0000$ achieved).
  * Operating decision threshold ($0.9953$) is selected from the ROC curve at $0.1\%$ FPR, never an arbitrary $0.5$ argmax.

### 2. Zero-Day & Novel Attack Detection
* **Problem**: Supervised models cannot detect attacks with feature signatures not present in the training set; novel attack variants get misclassified as benign.
* **Defensive Solution**:
  * An unsupervised **PyTorch Deep Autoencoder** ($22 \to 16 \to 8 \to 4 \to 8 \to 16 \to 22$) trained **exclusively on BENIGN flows**.
  * The bottleneck forces the network to compress normal traffic dynamics. When presented with held-out, completely unseen attacks (e.g. `PortScan`), the network fails to reconstruct the flow, generating an elevated Mean Squared Error ($MSE$).
  * Threshold calibrated at the 99th percentile of benign validation errors ($1.0\%$ benign FPR bound).
  * **Experiment Result**: While baseline Isolation Forest achieved only $9.49\%$ recall on the held-out class, the Autoencoder achieved **$100.00\%$ zero-day detection**.

### 3. Concept Drift & Model Governance
* **Problem**: Production network behavior changes over time, silently degrading classifier accuracy.
* **Defensive Solution**:
  * **Two-Sample Kolmogorov-Smirnov (KS) Test**: Per-feature statistical drift monitor comparing incoming flow distributions against the frozen baseline ($5,000$ training flows, $\alpha = 0.05$).
  * **Automated Promotion Gatekeeper**: Refuses candidate model deployment unless Recall @ 0.1% FPR strictly improves over the incumbent on the frozen held-out test split (Thursday–Friday).

### 4. SOC Alert Fatigue & Explainability Latency
* **Problem**: High-speed attacks (e.g., port scans) trigger thousands of alerts within seconds, overwhelming SOC analysts. Full SHAP calculation per flow adds prohibitive latency to scoring pipelines.
* **Defensive Solution**:
  * **Alert Aggregator**: Groups bursts by `(source_ip, destination_port_pattern, time_window)` within a sliding 60-second window. A $10,000$-probe port scan collapses into **one aggregated alert** with `hit_count = 10000`.
  * **Lazy On-Demand SHAP**: `shap.TreeExplainer` values are computed strictly on request when an analyst clicks an alert in the dashboard ($<15$ms response time), incurring zero scoring overhead during ingestion.

---

## Project Structure

```
├── api/
│   ├── aggregator.py          # Alert grouping & anti-fatigue engine
│   ├── database.py            # SQLAlchemy models (SQLite dev, Postgres ready)
│   ├── drift.py               # KS drift testing & promotion gatekeeper
│   ├── explainer.py           # Lazy SHAP TreeExplainer module
│   └── main.py                # FastAPI app with SSE stream and CSV/PCAP ingestion
├── data/
│   ├── label.py               # Time-ordered data generator & timetable joiner
│   └── processed/             # Parquet dataset (30,000 labeled flows)
├── experiments/
│   └── zero_day_experiment.py # Zero-day experiment holding out PortScan
├── features/
│   ├── extract.py             # Scapy flow parser & strict schema validator
│   └── schema_v1.json         # Frozen schema contract (22 features, 8 metadata)
├── frontend/
│   ├── src/App.jsx            # React + Vite SOC Cybersecurity Dashboard
│   ├── package.json           # Frontend dependencies (React 19, Vite, Lucide)
│   └── vite.config.js         # Vite configuration & FastAPI proxy
├── models/
│   ├── autoencoder.py         # PyTorch Autoencoder anomaly layer
│   ├── hybrid.py              # Decision fusion matrix (Critical/High/Medium/Low)
│   ├── isolation_forest.py    # Baseline anomaly detector
│   └── supervised_pipeline.pkl# Fitted LightGBM pipeline artifact
├── reports/
│   ├── data_quality_report.json
│   ├── phase2_metrics.json
│   ├── zero_day.md            # Zero-day experiment report
│   └── zero_day_metrics.json
├── tests/
│   ├── test_phase1.py         # Schema contract & Scapy flow extraction tests
│   ├── test_phase2.py         # Leak-free pipeline & threshold calibration tests
│   ├── test_phase3.py         # Autoencoder & hybrid score fusion tests
│   ├── test_phase4.py         # HTTP 422 rejection, aggregator & SHAP tests
│   └── test_phase5.py         # KS drift test & retrain gatekeeper tests
├── docker-compose.yml         # Container orchestration (API + Frontend)
└── Dockerfile.api             # FastAPI container specification
```

---

## How to Reproduce Every Number

### 1. Data Pipeline & Schema Validation
```bash
python -m data.label
```
* Emits `data/processed/flows_labeled.parquet` ($30,000$ flows, $3.23\%$ attack prevalence).
* Verifies zero missing values, time-ordered Mon–Fri temporal partition.

### 2. Train Supervised Model & Calibrate Threshold
```bash
python -m train.train_supervised
```
* Runs 15 Optuna cross-validation trials optimizing PR-AUC.
* Calibrates operating threshold at $0.1\%$ FPR ($0.9953$).
* Generates `reports/confusion_matrix.png`, `reports/pr_curve.png`, `reports/roc_curve.png`.

### 3. Run Zero-Day Experiment
```bash
python -m experiments.zero_day_experiment --held-out PortScan
```
* Purges `PortScan` from training split.
* Evaluates Supervised vs. PyTorch Autoencoder vs. Isolation Forest.
* Generates `reports/zero_day.md`.

### 4. Run the Full Test Suite
```bash
pytest tests/ -v
```
* Verifies all 15 tests across schema rejection, leakage guard, calibration math, aggregation, and drift.

### 5. Launch the System
```bash
# Terminal 1: Backend API
uvicorn api.main:app --reload --port 8000

# Terminal 2: React + Vite Frontend
cd frontend
npm run dev
```
Open `http://localhost:3000` to interact with the live SOC dashboard.

---

## Honest Limitations

1. **Synthetic Flow Ground Truth in Development**: While feature extraction is real (via Scapy parsing raw packets), testing without live multi-gigabit hardware replayed synthetic flow distributions mimicking CIC-IDS2017 timetable statistics. Real-world 10Gbps production links require kernel bypass (e.g. DPDK or PF_RING) rather than pure-Python socket parsers.
2. **Encrypted Payload Invisibility**: Flow-based NIDS inspects packet headers and statistical flow timing/sizes (inter-arrival times, byte rates, TCP flags), but cannot inspect encrypted TLS application payloads. Layer-7 attacks that blend into standard HTTPS burst patterns without altering packet counts may evade statistical flow detection.
3. **Autoencoder False Alarms on Novel Benign Workloads**: Training on benign traffic bounds false alarms under historical network patterns. However, legitimate administrative backups or new cloud sync services may initially yield elevated reconstruction error until incorporated into retrained reference baselines.
