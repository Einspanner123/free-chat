import { act } from "react";
import { describe, expect, it, vi } from "vitest";
import { MetricChart } from "../src/chart";
import { createRoot } from "react-dom/client";

const charts = vi.hoisted(() => ({
  first: { setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn() },
  second: { setOption: vi.fn(), resize: vi.fn(), dispose: vi.fn() },
  init: vi.fn(),
}));
vi.mock("echarts/core", () => ({
  use: vi.fn(), init: charts.init,
}));

describe("metric chart lifecycle", () => {
  it("updates labels and values, resizes the active chart, and releases resources on replacement/unmount", async () => {
    charts.init.mockReturnValueOnce(charts.first).mockReturnValueOnce(charts.second);
    const container = document.createElement("div");
    document.body.append(container);
    const root = createRoot(container);
    try {
      await act(async () => root.render(<MetricChart points={[{ label: "hit", value: 3 }]} />));
      expect(charts.init).toHaveBeenCalledWith(container.firstElementChild);
      window.dispatchEvent(new Event("resize"));
      expect(charts.first.resize).toHaveBeenCalledTimes(1);
      await act(async () => root.render(<MetricChart points={[{ label: "miss", value: 0 }]} />));
      expect(charts.first.dispose).toHaveBeenCalledTimes(1);
      expect(charts.second.setOption).toHaveBeenCalledWith(expect.objectContaining({
        xAxis: expect.objectContaining({ data: ["miss"] }),
        series: [expect.objectContaining({ data: [0] })],
      }));
      window.dispatchEvent(new Event("resize"));
      expect(charts.first.resize).toHaveBeenCalledTimes(1);
      expect(charts.second.resize).toHaveBeenCalledTimes(1);
    } finally {
      await act(async () => root.unmount());
      container.remove();
    }
    expect(charts.second.dispose).toHaveBeenCalledTimes(1);
    window.dispatchEvent(new Event("resize"));
    expect(charts.second.resize).toHaveBeenCalledTimes(1);
  });
});
