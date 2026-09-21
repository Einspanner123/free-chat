import { act, type ReactNode } from "react";
import { createRoot } from "react-dom/client";

export async function mount(node: ReactNode) {
  const container = document.createElement("div");
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => root.render(node));
  return {
    container,
    async unmount() {
      await act(async () => root.unmount());
      container.remove();
    },
  };
}

export async function settleUntil(check: () => void) {
  let lastError: unknown;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    await act(async () => new Promise((resolve) => setTimeout(resolve, 10)));
    try {
      check();
      return;
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}
