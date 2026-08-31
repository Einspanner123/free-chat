import { BarChart } from "echarts/charts";
import { GridComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { useEffect, useRef } from "react";

import type { MetricPoint } from "./api";

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

export function MetricChart({ points }: { points: MetricPoint[] }) {
  const element = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current);
    chart.setOption({
      animationDuration: 300,
      backgroundColor: "transparent",
      grid: { left: 44, right: 20, top: 24, bottom: 40 },
      xAxis: {
        type: "category",
        data: points.map((point) => point.label),
        axisLabel: { color: "#93a4b8", interval: 0 },
        axisLine: { lineStyle: { color: "#283848" } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#93a4b8" },
        splitLine: { lineStyle: { color: "#20303f" } },
      },
      series: [
        {
          type: "bar",
          data: points.map((point) => point.value),
          itemStyle: { color: "#5eead4", borderRadius: [5, 5, 0, 0] },
        },
      ],
      tooltip: { trigger: "axis" },
    });
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [points]);

  return <div className="chart" ref={element} aria-label="metric chart" />;
}
