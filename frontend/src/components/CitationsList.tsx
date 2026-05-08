type CitationsListProps = {
  citations?: string[];
};

export default function CitationsList({ citations = [] }: CitationsListProps) {
  return (
    <section className="result-card">
      <h2>Citations</h2>
      {citations.length === 0 ? (
        <p>No citations available yet.</p>
      ) : (
        <ul>
          {citations.map((citation) => (
            <li key={citation}>{citation}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
