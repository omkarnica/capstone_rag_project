import { ReactNode } from "react";

type DiagnosticItem = {
  label: string;
  value: string | number;
};

type DiagnosticsPanelProps = {
  items: DiagnosticItem[];
  onClearCache?: () => void;
  clearing?: boolean;
  controlsContent?: ReactNode;
};

export default function DiagnosticsPanel({
  items,
  onClearCache,
  clearing = false,
  controlsContent,
}: DiagnosticsPanelProps) {
  return (
    <aside className="diagnostics-panel">
      <section className="diagnostics-card">
        <h2>Run diagnostics</h2>
        <dl className="diagnostics-list">
          {items.map((item) => (
            <div key={item.label} className="diagnostics-row">
              <dt>{item.label}</dt>
              <dd>{item.value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="diagnostics-card diagnostics-controls">
        <h2>Controls</h2>
        {controlsContent ? (
          controlsContent
        ) : onClearCache ? (
          <button type="button" onClick={onClearCache} disabled={clearing}>
            {clearing ? "Clearing..." : "Clear cache"}
          </button>
        ) : null}
      </section>
    </aside>
  );
}
