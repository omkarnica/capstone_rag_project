import { useMemo, useState } from "react";
import AnswerPanel from "../components/AnswerPanel";
import CitationsList from "../components/CitationsList";
import ContextsList from "../components/ContextsList";
import DiagnosticsPanel from "../components/DiagnosticsPanel";
import ProjectTabs from "../components/ProjectTabs";
import QueryForm from "../components/QueryForm";
import StatusCard from "../components/StatusCard";
import { clearCache, QueryResult, runQuery } from "../lib/api";
import "../styles/workspace.css";

export default function WorkspacePage() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [queryLatencyMs, setQueryLatencyMs] = useState<number | null>(null);
  const [result, setResult] = useState<QueryResult | null>(null);

  const diagnostics = useMemo(
    () => [
      { label: "Decision", value: String(result?.final_route ?? "-") },
      { label: "Cache", value: String(result?.cache_status ?? "idle") },
      { label: "Response source", value: String(result?.response_source ?? "-") },
      { label: "Retrieved contexts", value: Number(result?.retrieved_context_count ?? 0) },
      { label: "Citations", value: Number(result?.citations?.length ?? 0) },
      { label: "Latency", value: queryLatencyMs === null ? "-" : `${queryLatencyMs} ms` },
    ],
    [queryLatencyMs, result],
  );

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    setQueryLatencyMs(null);

    try {
      const startedAt = performance.now();
      const nextResult = await runQuery({ question });
      setResult(nextResult);
      setQueryLatencyMs(Math.max(0, Math.round(performance.now() - startedAt)));
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : "Query failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleClearCache() {
    setClearing(true);
    try {
      await clearCache();
    } catch (nextError) {
      const message = nextError instanceof Error ? nextError.message : "Cache clear failed";
      setError(message);
    } finally {
      setClearing(false);
    }
  }

  return (
    <main className="workspace-page">
      <div className="workspace-shell">
        <DiagnosticsPanel items={diagnostics} onClearCache={handleClearCache} clearing={clearing} />

        <section className="workspace-main">
          <header className="workspace-header">
            <ProjectTabs />
            <div className="workspace-title">
              <p className="landing-eyebrow">SMART FINANCIAL RAG WORKSPACE</p>
              <h1>Smart Financial RAG</h1>
              <p>Run routed financial intelligence queries and inspect answer evidence in one place.</p>
            </div>

            <div className="workspace-status-grid">
              <StatusCard
                label="Upload Status"
                value="Filings | XBRL | Patent | Contradictions uploaded"
                detail="Backend services are live and ready for routed queries."
              />
              <StatusCard
                label="Model Route"
                value={String(result?.final_route ?? "-")}
                detail="SQL, graph, filings, transcripts, patents, or contradiction routes."
              />
              <StatusCard
                label="Cache Status"
                value={String(result?.cache_status ?? "idle")}
                detail={`Response source: ${String(result?.response_source ?? "-")}`}
              />
            </div>
          </header>

          <QueryForm
            question={question}
            loading={loading}
            onQuestionChange={setQuestion}
            onSubmit={handleSubmit}
          />

          <AnswerPanel answer={result?.final_answer} error={error} />
          <CitationsList citations={result?.citations} />
          <ContextsList contexts={result?.retrieved_contexts} />
        </section>
      </div>
    </main>
  );
}
