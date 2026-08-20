import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { execSync } from "node:child_process";
import pkg from "./package.json";

const host = process.env.TAURI_DEV_HOST;

// Short hash of the last commit, shown in the version badge.
const buildHash = (() => {
  try {
    return execSync("git rev-parse --short HEAD", { encoding: "utf8" }).trim();
  } catch {
    return "dev";
  }
})();

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  define: {
    __APP_VERSION__: JSON.stringify(pkg.version),
    __BUILD_HASH__: JSON.stringify(buildHash),
  },
  server: {
    port: 1420,
    strictPort: true,
    // Bind IPv4 loopback explicitly: WebKitGTK resolves localhost to
    // 127.0.0.1, while `false` let Vite bind only ::1 on some systems.
    host: host || "127.0.0.1",
    watch: { ignored: ["**/crates/**"] },
  },
  build: {
    target: "chrome105",
    minify: false,
    sourcemap: true,
  },
});
