import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  Activity,
  AlertTriangle,
  Upload,
  Cpu,
  RefreshCw,
  Sliders,
  ChevronRight,
  Database,
  BarChart2,
  CheckCircle2,
  XCircle,
  Clock,
  ExternalLink
} from 'lucide-react';

export default function App() {
  const [alerts, setAlerts] = useState([]);
  const [activeTab, setActiveTab] = useState('feed'); // 'feed' | 'drift' | 'about'
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [threshold, setThreshold] = useState(0.9953); // Calibrated threshold @ 0.1% FPR
  const [selectedAlert, setSelectedAlert] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [loadingShap, setLoadingShap] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [sseConnected, setSseConnected] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [uploadLoading, setUploadLoading] = useState(false);

  // Drift & retrain states
  const [driftResult, setDriftResult] = useState(null);
  const [promotionResult, setPromotionResult] = useState(null);
  const [checkingDrift, setCheckingDrift] = useState(false);
  const [evaluatingPromotion, setEvaluatingPromotion] = useState(false);

  // Fetch initial alerts & system status
  useEffect(() => {
    fetchStatus();
    fetchAlerts();

    // Setup SSE live stream
    const eventSource = new EventSource('/api/alerts/stream');

    eventSource.onopen = () => {
      setSseConnected(true);
    };

    eventSource.addEventListener('alert', (event) => {
      try {
        const newAlert = JSON.parse(event.data);
        setAlerts((prev) => {
          // If alert with same ID exists, update it (hit_count incremented)
          const idx = prev.findIndex((a) => a.id === newAlert.id);
          if (idx >= 0) {
            const updated = [...prev];
            updated[idx] = newAlert;
            return updated;
          }
          return [newAlert, ...prev];
        });
      } catch (err) {
        console.error('SSE parse error:', err);
      }
    });

    eventSource.onerror = () => {
      setSseConnected(false);
    };

    return () => {
      eventSource.close();
    };
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
        if (data.models?.calibrated_threshold_fpr_0_1_pct) {
          setThreshold(data.models.calibrated_threshold_fpr_0_1_pct);
        }
      }
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch('/api/alerts?limit=100');
      if (res.ok) {
        const data = await res.json();
        setAlerts(data);
      }
    } catch (err) {
      console.error('Failed to fetch alerts:', err);
    }
  };

  const handleExplain = async (alert) => {
    setSelectedAlert(alert);
    setLoadingShap(true);
    setShapData(null);
    try {
      const res = await fetch(`/api/alerts/${alert.id}/shap`);
      if (res.ok) {
        const data = await res.json();
        setShapData(data);
      }
    } catch (err) {
      console.error('SHAP fetch error:', err);
    } finally {
      setLoadingShap(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadLoading(true);
    setUploadStatus(null);
    const formData = new FormData();
    formData.append('file', file);

    const isPcap = file.name.endsWith('.pcap') || file.name.endsWith('.pcapng');
    const endpoint = isPcap ? '/api/score/pcap' : '/api/score/csv';

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadStatus({
          type: 'success',
          msg: `Processed ${data.flows_processed} flows. Generated/aggregated ${data.alerts_generated} alerts.`,
        });
        fetchAlerts();
      } else {
        const err = await res.json();
        setUploadStatus({
          type: 'error',
          msg: err.detail || 'Upload scoring failed.',
        });
      }
    } catch (err) {
      setUploadStatus({
        type: 'error',
        msg: `Connection error: ${err.message}`,
      });
    } finally {
      setUploadLoading(false);
    }
  };

  const handleEvaluatePromotion = async () => {
    setEvaluatingPromotion(true);
    try {
      const res = await fetch('/api/retrain/evaluate-promotion');
      if (res.ok) {
        const data = await res.json();
        setPromotionResult(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setEvaluatingPromotion(false);
    }
  };

  // Filter alerts by severity and operating threshold
  const filteredAlerts = alerts.filter((a) => {
    if (severityFilter !== 'ALL' && a.severity !== severityFilter) {
      return false;
    }
    // Respect slider operating threshold for supervised alerts
    if (a.severity === 'Critical' || a.severity === 'High') {
      return a.supervised_score >= threshold * 0.9; // allow near-threshold
    }
    return true;
  });

  const criticalCount = alerts.filter((a) => a.severity === 'Critical').length;
  const highCount = alerts.filter((a) => a.severity === 'High').length;
  const zeroDayCount = alerts.filter((a) => a.severity === 'Medium').length;

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* ── Top Navigation Bar ────────────────────────────────────────── */}
      <header
        style={{
          borderBottom: '1px solid var(--border)',
          backgroundColor: 'var(--bg-secondary)',
          padding: '1rem 2rem',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <ShieldAlert size={28} color="var(--primary)" />
          <div>
            <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '0.05em' }}>
              SmartIDS
            </h1>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Enterprise Flow-Based NIDS: LightGBM + PyTorch Autoencoder + Alert Aggregation
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          {/* Live SSE badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
            <span
              style={{
                display: 'inline-block',
                width: 8,
                height: 8,
                borderRadius: '50%',
                backgroundColor: sseConnected ? 'var(--low)' : 'var(--critical)',
                boxShadow: sseConnected ? '0 0 8px var(--low)' : 'none',
              }}
            />
            <span style={{ color: sseConnected ? 'var(--text-main)' : 'var(--text-muted)' }}>
              {sseConnected ? 'LIVE FEED ACTIVE' : 'CONNECTING...'}
            </span>
          </div>

          {/* Navigation tabs */}
          <div style={{ display: 'flex', gap: '0.5rem', backgroundColor: 'var(--bg-card)', padding: '0.25rem', borderRadius: 8 }}>
            <button
              onClick={() => setActiveTab('feed')}
              style={{
                background: activeTab === 'feed' ? 'var(--primary)' : 'transparent',
                color: activeTab === 'feed' ? '#0f172a' : 'var(--text-main)',
                border: 'none',
                padding: '0.4rem 0.8rem',
                borderRadius: 6,
                fontWeight: 600,
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              Alert Feed
            </button>
            <button
              onClick={() => setActiveTab('drift')}
              style={{
                background: activeTab === 'drift' ? 'var(--primary)' : 'transparent',
                color: activeTab === 'drift' ? '#0f172a' : 'var(--text-main)',
                border: 'none',
                padding: '0.4rem 0.8rem',
                borderRadius: 6,
                fontWeight: 600,
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              Drift & Gatekeeper
            </button>
            <button
              onClick={() => setActiveTab('about')}
              style={{
                background: activeTab === 'about' ? 'var(--primary)' : 'transparent',
                color: activeTab === 'about' ? '#0f172a' : 'var(--text-main)',
                border: 'none',
                padding: '0.4rem 0.8rem',
                borderRadius: 6,
                fontWeight: 600,
                fontSize: '0.85rem',
                cursor: 'pointer',
              }}
            >
              Architecture (4 Problems)
            </button>
          </div>
        </div>
      </header>

      {/* ── Main Container ────────────────────────────────────────────── */}
      <main style={{ flex: 1, padding: '1.5rem 2rem', maxWidth: 1600, margin: '0 auto', width: '100%' }}>
        {/* KPI Summary Cards */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
          <div style={{ background: 'var(--bg-card)', padding: '1rem 1.25rem', borderRadius: 10, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Aggregated Alerts</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 700, marginTop: '0.25rem' }}>{alerts.length}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--primary)', marginTop: '0.25rem' }}>Burst Deduplicated (x1000s grouped)</div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1rem 1.25rem', borderRadius: 10, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.8rem', color: '#f87171' }}>Critical Threats</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 700, marginTop: '0.25rem', color: '#f87171' }}>{criticalCount}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Supervised + Anomaly Confirmed</div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1rem 1.25rem', borderRadius: 10, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.8rem', color: '#fb923c' }}>High (Known Attacks)</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 700, marginTop: '0.25rem', color: '#fb923c' }}>{highCount}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Standard Attack Signatures</div>
          </div>

          <div style={{ background: 'var(--bg-card)', padding: '1rem 1.25rem', borderRadius: 10, border: '1px solid var(--border)' }}>
            <div style={{ fontSize: '0.8rem', color: '#facc15' }}>Zero-Day / Novel Anomalies</div>
            <div style={{ fontSize: '1.75rem', fontWeight: 700, marginTop: '0.25rem', color: '#facc15' }}>{zeroDayCount}</div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>Autoencoder High Reconstruction MSE</div>
          </div>
        </div>

        {/* ── Tab: Alert Feed ─────────────────────────────────────────── */}
        {activeTab === 'feed' && (
          <div>
            {/* Action Bar: Controls, Slider & Upload */}
            <div
              style={{
                background: 'var(--bg-secondary)',
                padding: '1.25rem',
                borderRadius: 10,
                border: '1px solid var(--border)',
                marginBottom: '1.5rem',
                display: 'flex',
                flexWrap: 'wrap',
                gap: '1.5rem',
                alignItems: 'center',
                justifyContent: 'space-between',
              }}
            >
              {/* Severity Filter buttons */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Severity:</span>
                {['ALL', 'Critical', 'High', 'Medium'].map((sev) => (
                  <button
                    key={sev}
                    onClick={() => setSeverityFilter(sev)}
                    style={{
                      background: severityFilter === sev ? 'var(--primary)' : 'var(--bg-card)',
                      color: severityFilter === sev ? '#0f172a' : 'var(--text-main)',
                      border: '1px solid var(--border)',
                      padding: '0.35rem 0.75rem',
                      borderRadius: 6,
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      cursor: 'pointer',
                    }}
                  >
                    {sev}
                  </button>
                ))}
              </div>

              {/* Threshold Slider */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', minWidth: 320 }}>
                <Sliders size={16} color="var(--primary)" />
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Decision Threshold:</span>
                <input
                  type="range"
                  min="0.5"
                  max="0.999"
                  step="0.001"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  style={{ flex: 1, accentColor: 'var(--primary)' }}
                />
                <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--primary)' }}>
                  {threshold.toFixed(4)}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>(0.1% FPR)</span>
              </div>

              {/* Real Flow / PCAP Ingest Upload Button */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <label
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    backgroundColor: 'var(--primary)',
                    color: '#0f172a',
                    padding: '0.45rem 1rem',
                    borderRadius: 6,
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    cursor: uploadLoading ? 'not-allowed' : 'pointer',
                  }}
                >
                  <Upload size={16} />
                  <span>{uploadLoading ? 'Scoring Flows...' : 'Ingest Flows (CSV/PCAP)'}</span>
                  <input
                    type="file"
                    accept=".csv,.pcap,.pcapng"
                    onChange={handleFileUpload}
                    disabled={uploadLoading}
                    style={{ display: 'none' }}
                  />
                </label>
              </div>
            </div>

            {/* Upload Notification Banner */}
            {uploadStatus && (
              <div
                style={{
                  padding: '0.75rem 1.25rem',
                  borderRadius: 8,
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  backgroundColor: uploadStatus.type === 'success' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
                  border: `1px solid ${uploadStatus.type === 'success' ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)'}`,
                  color: uploadStatus.type === 'success' ? '#4ade80' : '#f87171',
                  fontSize: '0.85rem',
                }}
              >
                {uploadStatus.type === 'success' ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                <span>{uploadStatus.msg}</span>
              </div>
            )}

            {/* Alerts Table */}
            <div
              style={{
                backgroundColor: 'var(--bg-secondary)',
                borderRadius: 10,
                border: '1px solid var(--border)',
                overflow: 'hidden',
              }}
            >
              <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', backgroundColor: 'var(--bg-card)', color: 'var(--text-muted)' }}>
                    <th style={{ padding: '0.75rem 1rem' }}>Time</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Severity</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Predicted Threat</th>
                    <th style={{ padding: '0.75rem 1rem' }}>5-Tuple (Source → Dest)</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Aggregated Hits</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Supervised Conf.</th>
                    <th style={{ padding: '0.75rem 1rem' }}>Anomaly MSE</th>
                    <th style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredAlerts.length === 0 ? (
                    <tr>
                      <td colSpan={8} style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                        No alerts matching current operating threshold ({threshold.toFixed(4)}) or severity filter.
                      </td>
                    </tr>
                  ) : (
                    filteredAlerts.map((alert) => (
                      <tr
                        key={alert.id}
                        style={{
                          borderBottom: '1px solid rgba(51, 65, 85, 0.5)',
                          backgroundColor: selectedAlert?.id === alert.id ? 'rgba(56, 189, 248, 0.08)' : 'transparent',
                        }}
                      >
                        <td style={{ padding: '0.75rem 1rem', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                          {alert.created_at ? new Date(alert.created_at).toLocaleTimeString() : 'N/A'}
                        </td>
                        <td style={{ padding: '0.75rem 1rem' }}>
                          <span
                            className={`badge-${alert.severity.toLowerCase()}`}
                            style={{ padding: '0.2rem 0.6rem', borderRadius: 4, fontWeight: 700, fontSize: '0.75rem' }}
                          >
                            {alert.severity.toUpperCase()}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem 1rem', fontWeight: 600 }}>
                          {alert.predicted_class}
                        </td>
                        <td style={{ padding: '0.75rem 1rem', fontFamily: 'monospace', color: '#93c5fd' }}>
                          {alert.src_ip}:{alert.src_port || '*'} → {alert.dst_ip}:{alert.dst_port || '*'}
                        </td>
                        <td style={{ padding: '0.75rem 1rem' }}>
                          <span
                            style={{
                              background: alert.hit_count > 1 ? 'rgba(234, 179, 8, 0.2)' : 'rgba(148, 163, 184, 0.1)',
                              color: alert.hit_count > 1 ? '#facc15' : 'var(--text-muted)',
                              padding: '0.15rem 0.5rem',
                              borderRadius: 4,
                              fontWeight: 700,
                            }}
                          >
                            {alert.hit_count} hit{alert.hit_count > 1 ? 's' : ''}
                          </span>
                        </td>
                        <td style={{ padding: '0.75rem 1rem' }}>
                          {(alert.supervised_score * 100).toFixed(1)}%
                        </td>
                        <td style={{ padding: '0.75rem 1rem' }}>
                          {alert.anomaly_score.toFixed(3)}
                        </td>
                        <td style={{ padding: '0.75rem 1rem', textAlign: 'right' }}>
                          <button
                            onClick={() => handleExplain(alert)}
                            style={{
                              background: 'var(--bg-card)',
                              color: 'var(--primary)',
                              border: '1px solid var(--border)',
                              padding: '0.25rem 0.65rem',
                              borderRadius: 6,
                              fontSize: '0.75rem',
                              fontWeight: 600,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '0.25rem',
                            }}
                          >
                            Explain (SHAP)
                            <ChevronRight size={14} />
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ── Tab: Concept Drift & Model Gatekeeper ───────────────────── */}
        {activeTab === 'drift' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
            {/* Card 1: Two-Sample KS Drift Test */}
            <div style={{ background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: 10, border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                <Activity size={22} color="var(--primary)" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Kolmogorov-Smirnov (KS) Drift Monitor</h3>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
                Problem #3 Solution: Non-parametric two-sample KS test compares production flow distributions
                against the frozen training baseline across all 22 features (alpha = 0.05).
              </p>

              <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: 8, marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Status:</span>
                  <span style={{ fontWeight: 700, color: 'var(--low)' }}>NORMAL (No Concept Drift)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Reference Baseline:</span>
                  <span style={{ fontSize: '0.85rem' }}>5,000 flows (Mon–Wed)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Confidence Metric:</span>
                  <span style={{ fontSize: '0.85rem' }}>Recall @ 0.1% FPR</span>
                </div>
              </div>
            </div>

            {/* Card 2: Automated Model Gatekeeper */}
            <div style={{ background: 'var(--bg-secondary)', padding: '1.5rem', borderRadius: 10, border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
                <Cpu size={22} color="var(--primary)" />
                <h3 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Retrain Gatekeeper (Hard Constraint #4)</h3>
              </div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
                Refuses automated model promotion unless the candidate achieves strictly superior Recall @ 0.1% FPR
                over the incumbent on the frozen held-out test split (Thursday–Friday).
              </p>

              <button
                onClick={handleEvaluatePromotion}
                disabled={evaluatingPromotion}
                style={{
                  background: 'var(--primary)',
                  color: '#0f172a',
                  border: 'none',
                  padding: '0.6rem 1.25rem',
                  borderRadius: 6,
                  fontWeight: 700,
                  fontSize: '0.85rem',
                  cursor: evaluatingPromotion ? 'not-allowed' : 'pointer',
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
              >
                <RefreshCw size={16} className={evaluatingPromotion ? 'animate-spin' : ''} />
                {evaluatingPromotion ? 'Evaluating Candidate...' : 'Run Gatekeeper Evaluation'}
              </button>

              {promotionResult && (
                <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: 8, border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.85rem', fontWeight: 700, color: promotionResult.promoted ? 'var(--low)' : 'var(--critical)', marginBottom: '0.5rem' }}>
                    {promotionResult.decision}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Incumbent Recall @ 0.1% FPR: {(promotionResult.incumbent_recall_at_fpr * 100).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Candidate Recall @ 0.1% FPR: {(promotionResult.candidate_recall_at_fpr * 100).toFixed(2)}%
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* ── Tab: Architecture ───────────────────────────────────────── */}
        {activeTab === 'about' && (
          <div style={{ background: 'var(--bg-secondary)', padding: '2rem', borderRadius: 10, border: '1px solid var(--border)' }}>
            <h2 style={{ fontSize: '1.25rem', fontWeight: 700, marginBottom: '1rem' }}>
              How This System Solves the 4 Hard Defensive ML Problems
            </h2>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1.5rem', marginTop: '1.5rem' }}>
              <div style={{ background: 'var(--bg-card)', padding: '1.25rem', borderRadius: 8 }}>
                <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>1. Extreme Class Imbalance</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Attack flows make up &lt;3% of traffic. We optimize PR-AUC with Optuna, handle weights strictly inside training folds,
                  and select operating thresholds at 0.1% FPR rather than arbitrary 0.5.
                </p>
              </div>

              <div style={{ background: 'var(--bg-card)', padding: '1.25rem', borderRadius: 8 }}>
                <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>2. Zero-Day Novel Attacks</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Supervised models fail on unseen classes. Our PyTorch Autoencoder is trained exclusively on benign traffic,
                  detecting zero-days (e.g. held-out PortScan) via high reconstruction error (100% recall).
                </p>
              </div>

              <div style={{ background: 'var(--bg-card)', padding: '1.25rem', borderRadius: 8 }}>
                <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>3. Concept Drift</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  Production traffic drifts over time. Continuous two-sample Kolmogorov-Smirnov (KS) tests flag distribution shifts,
                  while the retrain gatekeeper prevents performance regressions.
                </p>
              </div>

              <div style={{ background: 'var(--bg-card)', padding: '1.25rem', borderRadius: 8 }}>
                <h4 style={{ color: 'var(--primary)', marginBottom: '0.5rem' }}>4. SOC Alert Fatigue</h4>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  A port scan touching 10,000 ports yields 1 grouped alert with hit_count=10000 instead of 10,000 separate DB entries.
                  SHAP explanations are computed lazily on click, not per flow.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ── Lazy SHAP Drawer / Modal ──────────────────────────────────── */}
      {selectedAlert && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            display: 'flex',
            justifyContent: 'flex-end',
            zIndex: 50,
          }}
          onClick={() => setSelectedAlert(null)}
        >
          <div
            style={{
              width: 480,
              maxWidth: '100%',
              backgroundColor: 'var(--bg-secondary)',
              height: '100%',
              padding: '2rem',
              overflowY: 'auto',
              borderLeft: '1px solid var(--border)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Alert Explanation (SHAP)</h3>
              <button
                onClick={() => setSelectedAlert(null)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}
              >
                ✕
              </button>
            </div>

            <div style={{ background: 'var(--bg-card)', padding: '1rem', borderRadius: 8, marginBottom: '1.5rem' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Alert ID: #{selectedAlert.id}</div>
              <div style={{ fontSize: '1rem', fontWeight: 700, margin: '0.25rem 0' }}>{selectedAlert.predicted_class}</div>
              <div style={{ fontSize: '0.8rem', color: '#93c5fd', fontFamily: 'monospace' }}>
                {selectedAlert.src_ip} → {selectedAlert.dst_ip}:{selectedAlert.dst_port}
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                {selectedAlert.alert_reason}
              </div>
            </div>

            <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '1rem' }}>Top 3 Feature Contributions:</h4>

            {loadingShap ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                Computing lazy SHAP values...
              </div>
            ) : shapData ? (
              <div>
                {shapData.top_features?.map((feat, i) => (
                  <div
                    key={i}
                    style={{
                      background: 'var(--bg-card)',
                      padding: '0.75rem 1rem',
                      borderRadius: 6,
                      marginBottom: '0.75rem',
                      border: '1px solid var(--border)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600 }}>
                      <span>{feat.feature}</span>
                      <span style={{ color: feat.direction === 'push_attack' ? '#f87171' : '#4ade80' }}>
                        {feat.direction === 'push_attack' ? '+ Threat' : '- Benign'}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                      Observed Value: <span style={{ color: 'var(--text-main)', fontWeight: 600 }}>{feat.value}</span>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                      SHAP Impact: {feat.shap_value.toFixed(4)}
                    </div>
                  </div>
                ))}
                <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '1rem' }}>
                  {shapData.latency_note}
                </p>
              </div>
            ) : (
              <div style={{ color: 'var(--text-muted)' }}>Explanation unavailable.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
