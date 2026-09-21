import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config.ts";

export default mergeConfig(viteConfig, defineConfig({
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.{ts,tsx}"],
    setupFiles: ["./tests/setup.ts"],
    environmentOptions: { jsdom: { url: "http://localhost/", console: true } },
    coverage: {
      provider: "v8",
      include: ["src/**/*.{ts,tsx}"],
      reporter: ["text"],
      reportsDirectory: join(tmpdir(), `freechat-webui-coverage-${process.pid}`),
      thresholds: { statements: 80, branches: 80, functions: 80, lines: 80 },
    },
  },
}));
