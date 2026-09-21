import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import { ConsolePage } from "../src/page";
import { mount, settleUntil } from "./render";

const chart = vi.hoisted(() => ({ setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn() }));
vi.mock("echarts/core", () => ({ use: vi.fn(), init: vi.fn(() => chart) }));

async function page() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const view = await mount(
    <QueryClientProvider client={client}>
      <ConsolePage view="traces" title="Traces" eyebrow="Decision lineage" description="Placement evidence" />
    </QueryClientProvider>,
  );
  return { ...view, async unmount() { await view.unmount(); client.clear(); } };
}

describe("evidence view", () => {
  it("shows loading and the authenticated failure without claiming evidence", async () => {
    let reject!: (error: Error) => void;
    vi.stubGlobal("fetch", vi.fn(() => new Promise((_resolve, fail) => { reject = fail; })));
    const view = await page();
    try {
      expect(view.container.textContent).toContain("Loading evidence");
      await act(async () => reject(new Error("control plane returned 401")));
      await settleUntil(() => expect(view.container.textContent).toContain("control plane returned 401"));
      expect(view.container.querySelector(".claim")).toBeNull();
    } finally { await view.unmount(); }
  });

  it("renders unknown evidence with an explicit missing timestamp and empty sections", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503 })));
    const view = await page();
    try {
      await settleUntil(() => expect(view.container.textContent).toContain("UNVERIFIED"));
      expect(view.container.textContent).toContain("No observation timestamp");
      expect(view.container.querySelectorAll(".empty")).toHaveLength(2);
      expect(view.container.querySelector("table")).toBeNull();
    } finally { await view.unmount(); }
  });

  it("renders measured values and the union of record fields without losing false or zero", async () => {
    const snapshot = {
      state: "VERIFIED", summary: "Measured trace", generated_at: "2026-09-21T00:00:00Z",
      metrics: [{ label: "cache hits", value: 7 }],
      records: [{ id: "r1", count: 0, active: false, empty: null }, { name: "<unsafe>" }],
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(snapshot))));
    const view = await page();
    try {
      await settleUntil(() => expect(view.container.querySelector('[aria-label="metric chart"]')).not.toBeNull());
      expect(view.container.querySelector(".claim-verified")?.textContent).toBe("VERIFIED");
      expect(view.container.textContent).toContain("Observed 2026-09-21T00:00:00Z");
      expect(Array.from(view.container.querySelectorAll("th"), (cell) => cell.textContent))
        .toEqual(["id", "count", "active", "empty", "name"]);
      expect(Array.from(view.container.querySelectorAll("tbody tr:first-child td"), (cell) => cell.textContent))
        .toEqual(["r1", "0", "false", "—", "—"]);
      expect(view.container.textContent).toContain("<unsafe>");
      expect(view.container.querySelector("unsafe")).toBeNull();
      expect(chart.setOption).toHaveBeenCalledWith(expect.objectContaining({
        xAxis: expect.objectContaining({ data: ["cache hits"] }),
        series: [expect.objectContaining({ data: [7] })],
      }));
    } finally { await view.unmount(); }
  });
});
