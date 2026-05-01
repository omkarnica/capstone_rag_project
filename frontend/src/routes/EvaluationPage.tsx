import { useEffect, useMemo, useState } from "react";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import ProjectTabs from "../components/ProjectTabs";
import StatusCard from "../components/StatusCard";
import {
  getEvalRun,
  getEvalRunAblation,
  getEvalRunSummary,
  listEvalRuns,
  triggerEvalRun,
  type EvalRunAblationResponse,
  type EvalRunRecord,
  type EvalRunSummaryResponse,
} from "../lib/evalApi";
import "../styles/evaluation.css";
import "../styles/workspace.css";

const CONFIG_ORDER = ["naive_rag", "plus_router", "plus_reranker", "plus_kg", "full_system"];
const METRICS = ["faithfulness", "numerical_accuracy", "answer_relevancy", "due_diligence_confidence"];
const TIER_FILTERS = ["tier_1", "tier_2", "tier_3", "tier_4"] as const;

function formatMetricLabel(metric: string) {
  return metric.split("_").join(" ");
}

function getLastConfigName(run: EvalRunRecord | null): string {
  if (!run?.configs) {
    return "No runs yet";
  }

  const configs = Object.keys(run.configs);
  return configs.length ? configs[configs.length - 1] : "No runs yet";
}

function getNumericField(run: EvalRunRecord | null, keys: string[]): number | null {
  if (!run) {
    return null;
  }

  for (const key of keys) {
    const value = run[key];
    if (typeof value === "number") {
      return value;
    }
  }

  return null;
}

function averageMetric(summary: EvalRunSummaryResponse | null, configName: string, metric: string): number | null {
  const config = summary?.configs?.[configName];
  const value = config?.[metric];
  return typeof value === "number" ? value : null;
}

function getDeltaValue(
  ablation: EvalRunAblationResponse | null,
  configName: string,
  tier: string,
  metric: string,
): number | null {
  const delta = ablation?.baseline_delta?.[`${configName}_vs_naive`] ?? ablation?.baseline_delta?.[configName];
  const tierData = delta?.[tier];
  const value = tierData?.[metric];
  return typeof value === "number" ? value : null;
}

function formatDelta(value: number | null): string {
  if (value === null) {
    return "-";
  }
  return `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
}

function getDeltaClass(value: number | null): string {
  if (value === null || value === 0) {
    return "delta-neutral";
  }
  return value > 0 ? "delta-positive" : "delta-negative";
}

export default function EvaluationPage() {
  const [runIds, setRunIds] = useState<string[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [currentRun, setCurrentRun] = useState<EvalRunRecord | null>(null);
  const [summary, setSummary] = useState<EvalRunSummaryResponse | null>(null);
  const [ablation, setAblation] = useState<EvalRunAblationResponse | null>(null);
  const [selectedMetric, setSelectedMetric] = useState("faithfulness");
  const [selectedTier, setSelectedTier] = useState<(typeof TIER_FILTERS)[number]>("tier_1");
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const lastRunLabel = currentRun?.run_id ?? "No runs yet";
  const configsTestedCount = Object.keys(summary?.configs ?? currentRun?.configs ?? {}).length;
  const configsTestedLabel = configsTestedCount > 0 ? `${configsTestedCount} configs` : "5 configs";
  const datasetSizeLabel = "35 queries - 5 tiers";
  const hasRuns = runIds.length > 0;

  async function loadRun(runId: string) {
    if (!runId) {
      return;
    }

    setError(null);
    const run = await getEvalRun(runId);
    setCurrentRun(run);
    setSelectedRunId(runId);

    if (run.completed_at) {
      const [nextSummary, nextAblation] = await Promise.all([
        getEvalRunSummary(runId),
        getEvalRunAblation(runId),
      ]);
      setSummary(nextSummary);
      setAblation(nextAblation);
    } else {
      setSummary(null);
      setAblation(null);
    }
  }

  async function refreshRuns(selectLatest = true) {
    setLoadingHistory(true);
    try {
      const { run_ids } = await listEvalRuns();
      setRunIds(run_ids);

      if (!run_ids.length) {
        setSelectedRunId("");
        setCurrentRun(null);
        setSummary(null);
        setAblation(null);
        return;
      }

      const nextRunId = selectLatest ? run_ids[run_ids.length - 1] : selectedRunId || run_ids[0];
      if (nextRunId) {
        await loadRun(nextRunId);
      }
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : "Failed to load evaluation runs";
      setError(message);
    } finally {
      setLoadingHistory(false);
    }
  }

  useEffect(() => {
    void refreshRuns(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedRunId) {
      return undefined;
    }

    let active = true;
    let timer: number | undefined;

    const pollRun = async () => {
      try {
        setPolling(true);
        const run = await getEvalRun(selectedRunId);
        if (!active) {
          return;
        }
        setCurrentRun(run);

        if (run.completed_at) {
          const [nextSummary, nextAblation] = await Promise.all([
            getEvalRunSummary(selectedRunId),
            getEvalRunAblation(selectedRunId),
          ]);
          if (active) {
            setSummary(nextSummary);
            setAblation(nextAblation);
          }
        }
      } catch (nextError) {
        const message = nextError instanceof Error ? nextError.message : "Failed to poll evaluation run";
        if (active) {
          setError(message);
        }
      } finally {
        if (active) {
          setPolling(false);
        }
      }
    };

    void pollRun();
    timer = window.setInterval(() => {
      void pollRun();
    }, 10_000);

    return () => {
      active = false;
      if (timer) {
        window.clearInterval(timer);
      }
    };
  }, [selectedRunId]);

  async function handleRunEvaluation() {
    setTriggering(true);
    setError(null);

    try {
      const { run_id } = await triggerEvalRun();
      await refreshRuns(false);
      await loadRun(run_id);
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : "Failed to trigger evaluation";
      setError(message);
    } finally {
      setTriggering(false);
    }
  }

  const chartRows = useMemo(() => {
    return CONFIG_ORDER.map((configName) => ({
      configName,
      value: averageMetric(summary, configName, selectedMetric),
    }));
  }, [selectedMetric, summary]);

  const maxChartValue = Math.max(1, ...chartRows.map((row) => row.value ?? 0));
  const diagnosticsItems = useMemo(() => {
    const contextAverage = getNumericField(currentRun, ["avg_retrieved_contexts", "retrieved_contexts_avg"]);
    const latencyAverage = getNumericField(currentRun, ["avg_latency_ms", "average_latency_ms"]);
    const completedLabel = currentRun?.completed_at ? "Complete" : selectedRunId ? "Running" : "Idle";

    return [
      { label: "Decision", value: getLastConfigName(currentRun) },
      { label: "Retrieved contexts", value: typeof contextAverage === "number" ? contextAverage.toFixed(1) : "-" },
      { label: "Latency", value: typeof latencyAverage === "number" ? `${Math.round(latencyAverage)} ms` : "-" },
      { label: "Status", value: completedLabel },
      { label: "Poll", value: polling ? "Polling..." : selectedRunId ? "10s" : "-" },
    ];
  }, [currentRun, polling, selectedRunId]);

  return (
    <main className="evaluation-page">
      <div className="evaluation-shell">
        <DiagnosticsPanel
          items={diagnosticsItems}
          controlsContent={
            <div className="evaluation-controls-stack">
              <div className="evaluation-run-meta">
                <span className="evaluation-run-meta__label">Run ID</span>
                <span className="evaluation-run-meta__value">{currentRun?.run_id ?? "No runs yet"}</span>
                <span
                  className={
                    currentRun?.completed_at
                      ? "evaluation-badge complete"
                      : selectedRunId
                        ? "evaluation-badge pending"
                        : "evaluation-badge idle"
                  }
                >
                  {currentRun?.completed_at ? "Complete" : selectedRunId ? "Running" : "Idle"}
                </span>
              </div>

              <button type="button" onClick={handleRunEvaluation} disabled={triggering}>
                {triggering ? (
                  <>
                    <span className="evaluation-spinner" aria-hidden="true" />
                    Running...
                  </>
                ) : (
                  "Run Evaluation"
                )}
              </button>

              <label className="evaluation-select-field">
                <span>Past Runs</span>
                <select
                  value={selectedRunId}
                  onChange={(event) => void loadRun(event.target.value)}
                  disabled={loadingHistory || !runIds.length}
                >
                  {loadingHistory ? (
                    <option value="">Loading...</option>
                  ) : hasRuns ? (
                    <>
                      <option value="">Select a run</option>
                      {runIds.map((runId) => (
                        <option key={runId} value={runId}>
                          {runId}
                        </option>
                      ))}
                    </>
                  ) : (
                    <option value="">No runs available yet</option>
                  )}
                </select>
                <span className="evaluation-select-help">
                  {hasRuns
                    ? "Choose a previous evaluation to inspect its summary and deltas."
                    : "Trigger an evaluation run to populate this list."}
                </span>
              </label>
            </div>
          }
        />

        <section className="evaluation-main">
          <header className="workspace-header">
            <ProjectTabs />
            <div className="workspace-title">
              <p className="landing-eyebrow">SMART FINANCIAL RAG EVALUATION</p>
              <h1>Evaluation</h1>
              <p>Run ablation sweeps, inspect metric deltas, and compare system variants in one place.</p>
            </div>

            <div className="workspace-status-grid">
              <StatusCard label="Last Run" value={lastRunLabel} detail="Latest evaluation run identifier." />
              <StatusCard label="Configs Tested" value={configsTestedLabel} detail="naive_rag -> full_system." />
              <StatusCard label="Dataset Size" value={datasetSizeLabel} detail="Golden queries and tier coverage." />
            </div>
          </header>

          <section className="query-card evaluation-chart-card">
            <div className="evaluation-card-header">
              <div>
                <h2>Ablation Summary</h2>
                <p>Average score across the selected metric for each config.</p>
              </div>

              <label className="evaluation-select-field evaluation-metric-select">
                <span>Metric</span>
                <select value={selectedMetric} onChange={(event) => setSelectedMetric(event.target.value)}>
                  {METRICS.map((metric) => (
                    <option key={metric} value={metric}>
                      {formatMetricLabel(metric)}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="evaluation-chart">
              <div className="evaluation-chart-scale">
                <span>1.0</span>
                <span>0.5</span>
                <span>0.0</span>
              </div>

              <div className="evaluation-bars" aria-label="Ablation summary chart">
                {chartRows.map((row) => {
                  const value = row.value ?? 0;
                  const height = `${Math.max(6, Math.round((value / maxChartValue) * 100))}%`;
                  return (
                    <div key={row.configName} className="evaluation-bar">
                      <div className="evaluation-bar-track">
                        <div className="evaluation-bar-fill" style={{ height }} />
                      </div>
                      <div className="evaluation-bar-labels">
                        <span className="evaluation-bar-name">{row.configName}</span>
                        <span className="evaluation-bar-value">{row.value === null ? "-" : value.toFixed(2)}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </section>

          <section className="result-card evaluation-delta-card">
            <div className="evaluation-card-header">
              <div>
                <h2>Delta Table</h2>
                <p>Delta vs naive_rag baseline for each tier and metric.</p>
              </div>

              <div className="evaluation-tier-tabs" role="tablist" aria-label="Tier filter">
                {TIER_FILTERS.map((tier) => (
                  <button
                    key={tier}
                    type="button"
                    className={tier === selectedTier ? "evaluation-tier-tab is-active" : "evaluation-tier-tab"}
                    onClick={() => setSelectedTier(tier)}
                  >
                    {tier.replace("_", " ").toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            <div className="evaluation-table-wrap">
              <table className="evaluation-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    {["plus_router", "plus_reranker", "plus_kg", "full_system"].map((configName) => (
                      <th key={configName}>{configName}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map((metric) => (
                    <tr key={metric}>
                      <th>{formatMetricLabel(metric)}</th>
                      {["plus_router", "plus_reranker", "plus_kg", "full_system"].map((configName) => {
                        const value = getDeltaValue(ablation, configName, selectedTier, metric);
                        return (
                          <td key={`${configName}-${metric}`} className={getDeltaClass(value)}>
                            {formatDelta(value)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {error ? <p className="answer-card__error">{error}</p> : null}
          </section>
        </section>
      </div>
    </main>
  );
}
