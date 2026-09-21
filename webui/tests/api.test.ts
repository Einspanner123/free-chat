import { describe, expect, it, vi } from "vitest";
import { loadSnapshot } from "../src/api";

describe("control-plane snapshots", () => {
  it("forwards only the session credential and returns the server snapshot", async () => {
    const snapshot = { state: "VERIFIED", metrics: [], records: [], generated_at: null, summary: "live" };
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(snapshot)));
    vi.stubGlobal("fetch", fetch);
    sessionStorage.setItem("freechat.apiKey", "session-key");
    expect(await loadSnapshot("topology")).toEqual(snapshot);
    expect(fetch).toHaveBeenCalledWith("/control/ui/topology", {
      headers: { accept: "application/json", "x-api-key": "session-key" },
    });
  });

  it.each([404, 503])("reports missing evidence rather than verified data for HTTP %s", async (status) => {
    const fetch = vi.fn().mockResolvedValue(new Response(null, { status }));
    vi.stubGlobal("fetch", fetch);
    expect(await loadSnapshot("traces")).toMatchObject({
      state: "UNVERIFIED", metrics: [], records: [], generated_at: null,
    });
    expect(fetch).toHaveBeenCalledWith("/control/ui/traces", {
      headers: { accept: "application/json" },
    });
  });

  it.each([401, 403, 500])("does not hide HTTP %s errors as empty evidence", async (status) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status })));
    await expect(loadSnapshot("kv-cache")).rejects.toThrow(`control plane returned ${status}`);
  });

  it("propagates network and malformed JSON failures", async () => {
    const fetch = vi.fn().mockRejectedValueOnce(new Error("network unavailable"))
      .mockResolvedValueOnce(new Response("not JSON"));
    vi.stubGlobal("fetch", fetch);
    await expect(loadSnapshot("benchmarks")).rejects.toThrow("network unavailable");
    await expect(loadSnapshot("benchmarks")).rejects.toThrow();
  });
});
