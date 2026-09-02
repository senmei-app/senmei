// Postinstall: ensure the `cargo-tauri` CLI is present (Senmei pins the
// crates.io 2.x line). Skips fast when an installed 2.x already matches.
import { spawnSync } from "node:child_process";

const check = spawnSync("cargo", ["tauri", "--version"], { encoding: "utf8" });
if (check.error) {
  console.warn("cargo not found — skipping tauri-cli install");
  process.exit(0);
}
// A 1.x tauri-cli reports success too — only a 2.x version may short-circuit.
const major =
  check.status === 0
    ? /tauri-cli\s+(\d+)\./.exec(check.stdout ?? "")?.[1]
    : undefined;
if (major === "2") process.exit(0);
if (major !== undefined) {
  console.warn(`tauri-cli ${major}.x installed — need 2.x, installing`);
}

const install = spawnSync("cargo", ["install", "tauri-cli", "--version", "^2"], {
  stdio: "inherit",
});
process.exit(install.status ?? 1);
