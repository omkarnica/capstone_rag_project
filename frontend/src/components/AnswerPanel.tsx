type AnswerPanelProps = {
  answer?: string;
  error?: string | null;
};

export default function AnswerPanel({ answer, error }: AnswerPanelProps) {
  return (
    <section className="answer-card">
      <h2>Final answer</h2>
      {error ? <p className="answer-card__error">{error}</p> : null}
      <p>{answer || "Run a query to see the answer."}</p>
    </section>
  );
}
