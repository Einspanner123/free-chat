import type { EventEmitter } from "node:events";
import { act } from "react";
import { expect, it, vi } from "vitest";
import type { Root } from "react-dom/client";
import { settleUntil } from "./render";

const mounted = vi.hoisted(() => ({ root: null as Root | null }));
vi.mock("react-dom/client", async (importOriginal) => {
  const original = await importOriginal<typeof import("react-dom/client")>();
  return { ...original, createRoot: (...args: Parameters<typeof original.createRoot>) => {
    mounted.root = original.createRoot(...args);
    return mounted.root;
  } };
});

it("boots the actual router, navigates all five views, and applies/clears session-only credentials", async () => {
  vi.spyOn(window, "scrollTo").mockImplementation(() => {});
  // jsdom cannot reload Documents. Assert those two expected navigation calls,
  // while failing on any other DOM error instead of hiding application failures.
  const domConsole = (globalThis as typeof globalThis & {
    jsdom: { virtualConsole: EventEmitter };
  }).jsdom.virtualConsole;
  const originalListeners = domConsole.listeners("jsdomError");
  const browserErrors: Error[] = [];
  const captureError = (error: Error) => browserErrors.push(error);
  domConsole.removeAllListeners("jsdomError");
  domConsole.on("jsdomError", captureError);
  const fetch = vi.fn().mockImplementation(async () => new Response(null, { status: 404 }));
  vi.stubGlobal("fetch", fetch);
  const container = document.createElement("div");
  container.id = "root";
  document.body.append(container);
  try {
    await act(async () => { await import("../src/main"); });
    await settleUntil(() => expect(container.querySelector("h1")?.textContent).toBe("Topology"));
    expect(container.querySelectorAll("nav a")).toHaveLength(5);
    for (const title of ["Traces", "KV Cache", "Benchmarks", "Playground", "Topology"]) {
      const link = Array.from(container.querySelectorAll("nav a")).find((item) => item.textContent === title)!;
      await act(async () => link.dispatchEvent(new MouseEvent("click", { bubbles: true, button: 0 })));
      await settleUntil(() => expect(container.querySelector("h1")?.textContent).toBe(title));
    }
    expect(fetch.mock.calls.map(([url]) => url)).toEqual(expect.arrayContaining([
      "/control/ui/topology", "/control/ui/traces", "/control/ui/kv-cache",
      "/control/ui/benchmarks", "/control/ui/playground",
    ]));
    const input = container.querySelector<HTMLInputElement>("#api-key")!;
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(input, "new-key");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => container.querySelector<HTMLButtonElement>("button")!.click());
    expect(sessionStorage.getItem("freechat.apiKey")).toBe("new-key");
    expect(localStorage.getItem("freechat.apiKey")).toBeNull();
    await act(async () => {
      setValue.call(input, "");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => container.querySelector<HTMLButtonElement>("button")!.click());
    expect(sessionStorage.getItem("freechat.apiKey")).toBeNull();
    expect(browserErrors).toHaveLength(2);
    for (const error of browserErrors) {
      expect(String(error)).toContain("Not implemented: navigation to another Document");
    }
  } finally {
    await act(async () => mounted.root?.unmount());
    container.remove();
    domConsole.removeListener("jsdomError", captureError);
    for (const listener of originalListeners) domConsole.on("jsdomError", listener as (error: Error) => void);
  }
});
