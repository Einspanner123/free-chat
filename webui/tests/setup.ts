import { afterEach, vi } from "vitest";

// React's act boundary also covers QueryClient and router asynchronous updates.
Object.assign(globalThis, { IS_REACT_ACT_ENVIRONMENT: true });
afterEach(() => {
  sessionStorage.clear();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
