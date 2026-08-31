export type EvidenceState = "VERIFIED" | "UNVERIFIED" | "BLOCKED";

export interface MetricPoint {
  label: string;
  value: number;
}

export interface ConsoleSnapshot {
  state: EvidenceState;
  generated_at: string | null;
  summary: string;
  metrics: MetricPoint[];
  records: Array<Record<string, string | number | boolean | null>>;
}

const emptySnapshot: ConsoleSnapshot = {
  state: "UNVERIFIED",
  generated_at: null,
  summary: "No control-plane evidence has been returned for this view.",
  metrics: [],
  records: [],
};

export async function loadSnapshot(view: string): Promise<ConsoleSnapshot> {
  const apiKey = sessionStorage.getItem("freechat.apiKey");
  const response = await fetch(`/control/ui/${view}`, {
    headers: {
      accept: "application/json",
      ...(apiKey ? { "x-api-key": apiKey } : {}),
    },
  });
  if (response.status === 404 || response.status === 503) {
    return emptySnapshot;
  }
  if (!response.ok) {
    throw new Error(`control plane returned ${response.status}`);
  }
  return (await response.json()) as ConsoleSnapshot;
}
