import { useQuery } from "@tanstack/react-query";
import { lazy, Suspense } from "react";

import { loadSnapshot } from "./api";

const MetricChart = lazy(() =>
  import("./chart").then((module) => ({ default: module.MetricChart })),
);

interface PageProps {
  view: string;
  title: string;
  eyebrow: string;
  description: string;
}

export function ConsolePage({ view, title, eyebrow, description }: PageProps) {
  const query = useQuery({
    queryKey: ["console", view],
    queryFn: () => loadSnapshot(view),
    refetchInterval: 10_000,
  });

  if (query.isPending) return <PageMessage title={title} message="Loading evidence…" />;
  if (query.isError) return <PageMessage title={title} message={query.error.message} />;

  const snapshot = query.data;
  return (
    <main>
      <header className="page-header">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p className="description">{description}</p>
        </div>
        <span className={`claim claim-${snapshot.state.toLowerCase()}`}>{snapshot.state}</span>
      </header>

      <section className="panel summary-panel">
        <p>{snapshot.summary}</p>
        <small>{snapshot.generated_at ? `Observed ${snapshot.generated_at}` : "No observation timestamp"}</small>
      </section>

      <section className="panel">
        <div className="panel-title">Measured signals</div>
        {snapshot.metrics.length ? (
          <Suspense fallback={<div className="empty">Loading chart…</div>}>
            <MetricChart points={snapshot.metrics} />
          </Suspense>
        ) : <Empty />}
      </section>

      <section className="panel">
        <div className="panel-title">Evidence records</div>
        {snapshot.records.length ? <RecordTable records={snapshot.records} /> : <Empty />}
      </section>
    </main>
  );
}

function PageMessage({ title, message }: { title: string; message: string }) {
  return (
    <main>
      <h1>{title}</h1>
      <section className="panel">{message}</section>
    </main>
  );
}

function Empty() {
  return <div className="empty">No verified records available.</div>;
}

function RecordTable({ records }: { records: Array<Record<string, unknown>> }) {
  const columns = Array.from(new Set(records.flatMap((record) => Object.keys(record))));
  return (
    <div className="table-scroll">
      <table>
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>
          {records.map((record, index) => (
            <tr key={String(record.id ?? index)}>
              {columns.map((column) => <td key={column}>{String(record[column] ?? "—")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
