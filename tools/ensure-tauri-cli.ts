// Postinstall: ensure the `cargo-tauri` CLI is present (Senmei pins the
// crates.io 2.x line). Skips fast when an installed binary already matches.
import { spawnSync } from "node:child_process";

const check = spawnSync("cargo", ["tauri", "--version"], { stdio: "ignore" });
if (check.error) {
  console.warn("cargo not found — skipping tauri-cli install");
  process.exit(0);
}
if (check.status === 0) process.exit(0);

const install = spawnSync("cargo", ["install", "tauri-cli", "--version", "^2"], {
  stdio: "inherit",
});
process.exit(install.status ?? 1);
