type StatusCardProps = {
  label: string;
  value: string;
  detail?: string;
};

export default function StatusCard({ label, value, detail }: StatusCardProps) {
  return (
    <article className="status-card">
      <p className="status-card__label">{label}</p>
      <h3 className="status-card__value">{value}</h3>
      {detail ? <p className="status-card__detail">{detail}</p> : null}
    </article>
  );
}
