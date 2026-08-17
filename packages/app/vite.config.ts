import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
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
