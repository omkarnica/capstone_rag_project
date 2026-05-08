type QueryFormProps = {
  question: string;
  loading: boolean;
  onQuestionChange: (value: string) => void;
  onSubmit: () => void;
};

export default function QueryForm({
  question,
  loading,
  onQuestionChange,
  onSubmit,
}: QueryFormProps) {
  return (
    <section className="query-card">
      <h2>Ask a question</h2>
      <p>Enter a financial or filing question, then run the routed retrieval flow.</p>
      <textarea
        value={question}
        onChange={(event) => onQuestionChange(event.target.value)}
        placeholder="Type your query here..."
        rows={7}
      />
      <button type="button" onClick={onSubmit} disabled={loading || !question.trim()}>
        {loading ? "Running..." : "Run query"}
      </button>
    </section>
  );
}
