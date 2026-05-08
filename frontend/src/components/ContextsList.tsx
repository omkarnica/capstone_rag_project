type ContextsListProps = {
  contexts?: string[];
};

export default function ContextsList({ contexts = [] }: ContextsListProps) {
  return (
    <section className="result-card">
      <h2>Retrieved contexts</h2>
      {contexts.length === 0 ? (
        <p>No retrieved contexts available yet.</p>
      ) : (
        <ul className="contexts-list">
          {contexts.map((context) => (
            <li key={context}>
              <pre>{context}</pre>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
