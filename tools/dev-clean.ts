// Dev-clean: kill the Vite dev server (1420) + running Senmei, then clear the
// stale webview cache. Cross-platform (Linux/macOS/Windows incl. Wine).
import { execSync, spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

const win = process.platform === "win32";

function quiet(cmd: string) {
  try {
    execSync(cmd, { stdio: "ignore" });
  } catch {
    // nothing running / tool missing — fine
  }
}

// PowerShell pipelines need an explicit -Command, not the cmd-style default.
function quietPwsh(script: string) {
  try {
    spawnSync("powershell", ["-NoProfile", "-NonInteractive", "-Command", script], {
      stdio: "ignore",
    });
  } catch {
    // nothing running / tool missing — fine
  }
}

// Kill the Vite dev server on 1420.
if (win) {
  quietPwsh(
    "Get-NetTCPConnection -LocalPort 1420 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }",
  );
} else {
  quiet("lsof -ti:1420 | xargs kill -9");
}

// Kill the running app (best effort; not running is fine).
if (win) {
  quiet("taskkill /IM senmei.exe /F");
  quiet("taskkill /IM senmei-server.exe /F");
} else {
  quiet("pkill -x senmei; pkill -x senmei-server");
}

// Clear the platform's Tauri webview cache (identifier-based dir).
const id = "app.senmei.desktop";
let base: string;
if (win) {
  base = path.join(process.env.LOCALAPPDATA ?? homedir(), id);
} else if (process.platform === "darwin") {
  base = path.join(homedir(), "Library", "WebKit", id);
} else {
  base = path.join(
    process.env.XDG_DATA_HOME ?? path.join(homedir(), ".local", "share"),
    id,
  );
}
for (const name of [
  "WebKitCache",
  "CacheStorage",
  "hsts-storage.sqlite",
  "storage",
  "mediakeys",
  "WebsiteData",
  "EBWebView",
  "GPUCache",
  "Code Cache",
]) {
  const p = path.join(base, name);
  if (existsSync(p)) rmSync(p, { recursive: true, force: true });
}
console.log("cleaned");
