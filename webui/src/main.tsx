import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  Link,
  Outlet,
  RouterProvider,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";

import { ConsolePage } from "./page";
import "./styles.css";

const views = [
  ["topology", "Topology", "Physical reality", "GPU, NUMA, fabric and network observations."],
  ["traces", "Traces", "Decision lineage", "Agent lifecycle, placement and failure spans."],
  ["kv-cache", "KV Cache", "Memory lifecycle", "Retain, offload, prefetch and eviction evidence."],
  ["benchmarks", "Benchmarks", "Claims gate", "Baselines, confidence intervals and guardrails."],
  ["playground", "Playground", "Protocol surface", "Exercise compatible APIs with explicit Agent hints."],
] as const;

function Shell() {
  const [apiKey, setApiKey] = useState(() => sessionStorage.getItem("freechat.apiKey") ?? "");
  const saveApiKey = () => {
    if (apiKey) sessionStorage.setItem("freechat.apiKey", apiKey);
    else sessionStorage.removeItem("freechat.apiKey");
    window.location.reload();
  };
  return (
    <div className="shell">
      <aside>
        <div className="brand"><span>FC</span><strong>FreeChat</strong></div>
        <p className="aside-label">Infrastructure console</p>
        <nav>{views.map(([path, title]) => <Link key={path} to={`/${path}`} activeProps={{ className: "active" }}>{title}</Link>)}</nav>
        <div className="credential">
          <label htmlFor="api-key">Session API key</label>
          <input id="api-key" type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} />
          <button type="button" onClick={saveApiKey}>Apply</button>
        </div>
        <div className="aside-foot">Claims require archived evidence.</div>
      </aside>
      <Outlet />
    </div>
  );
}

const rootRoute = createRootRoute({ component: Shell });
const routes = views.map(([path, title, eyebrow, description], index) =>
  createRoute({
    getParentRoute: () => rootRoute,
    path: index === 0 ? "/" : `/${path}`,
    component: () => <ConsolePage view={path} title={title} eyebrow={eyebrow} description={description} />,
  }),
);
const topologyAlias = createRoute({
  getParentRoute: () => rootRoute,
  path: "/topology",
  component: () => <ConsolePage view="topology" title="Topology" eyebrow="Physical reality" description="GPU, NUMA, fabric and network observations." />,
});
const router = createRouter({ routeTree: rootRoute.addChildren([...routes, topologyAlias]) });
const queryClient = new QueryClient();

declare module "@tanstack/react-router" {
  interface Register { router: typeof router }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode><QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider></StrictMode>,
);
