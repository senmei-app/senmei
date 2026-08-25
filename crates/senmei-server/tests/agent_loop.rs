//! E2E agent loop over MCP stdio: probe → render_sample → compare_sample →
//! propose_render → confirm_render → poll status → assert output.
//!
//! Run: cargo test -p senmei-server --features render --test agent_loop -- --ignored --nocapture
//! Needs: Vulkan, a converted `fallin-soft` `.bpk`, and ffmpeg in PATH.

#![cfg(feature = "render")]

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

struct Client {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: u64,
}

impl Client {
    fn spawn() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_senmei-server"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn senmei-server");
        Self {
            stdin: child.stdin.take().unwrap(),
            reader: BufReader::new(child.stdout.take().unwrap()),
            child,
            next_id: 1,
        }
    }

    fn request(&mut self, method: &str, params: serde_json::Value) -> serde_json::Value {
        let id = self.next_id;
        self.next_id += 1;
        let req = serde_json::json!({"jsonrpc":"2.0","id":id,"method":method,"params":params});
        writeln!(self.stdin, "{req}").unwrap();
        self.stdin.flush().unwrap();
        loop {
            let mut line = String::new();
            self.reader.read_line(&mut line).unwrap();
            if line.trim().is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(&line).expect("json response");
            if v.get("id").and_then(serde_json::Value::as_u64) == Some(id) {
                return v;
            }
        }
    }

    fn notify(&mut self, method: &str) {
        writeln!(
            self.stdin,
            "{}",
            serde_json::json!({"jsonrpc":"2.0","method":method})
        )
        .unwrap();
        self.stdin.flush().unwrap();
    }

    fn call_tool(&mut self, name: &str, args: serde_json::Value) -> serde_json::Value {
        let r = self.request(
            "tools/call",
            serde_json::json!({"name": name, "arguments": args}),
        );
        let text = r["result"]["content"][0]["text"]
            .as_str()
            .expect("tool text");
        serde_json::from_str(text).unwrap_or_else(|_| serde_json::json!({ "raw": text }))
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
#[ignore = "needs Vulkan + converted fallin-soft .bpk + ffmpeg in PATH"]
fn agent_loop_probe_sample_compare_render() {
    let dir = std::env::temp_dir().join("senmei-agent-loop");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.mp4");
    let ok = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30",
            "-t",
            "1",
            "-y",
        ])
        .arg(&input)
        .status()
        .expect("run ffmpeg");
    assert!(ok.success(), "ffmpeg input generation failed");

    let input_s = input.to_str().unwrap();
    let mut c = Client::spawn();
    c.request(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "agent-loop-test", "version": "0"}
        }),
    );
    c.notify("notifications/initialized");

    // 1. probe
    let probe = c.call_tool("probe_video", serde_json::json!({"input": input_s}));
    assert_eq!(probe["width"], 320);
    assert_eq!(probe["height"], 240);

    // 2. sample render (no confirm gate)
    let sample = dir.join("sample.mp4");
    let sample_s = sample.to_str().unwrap();
    let got = c.call_tool(
        "render_sample",
        serde_json::json!({
            "input": input_s,
            "output": sample_s,
            "scale": 2,
            "modelId": "fallin-soft",
            "startMs": 0,
            "endMs": 1000
        }),
    );
    assert_eq!(got["output"], sample_s);
    assert!(sample.exists(), "sample missing");

    // 3. compare vs source
    let cmp = c.call_tool(
        "compare_sample",
        serde_json::json!({"original": input_s, "rendered": sample_s}),
    );
    assert!(
        cmp["psnrDb"].is_number() && cmp["ssim"].is_number(),
        "no metrics: {cmp}"
    );

    // 4. propose + confirm full render
    let out = dir.join("out.mp4");
    let out_s = out.to_str().unwrap();
    let propose = c.call_tool(
        "propose_render",
        serde_json::json!({"input": input_s, "output": out_s, "scale": 2, "modelId": "fallin-soft"}),
    );
    assert_eq!(propose, "render proposed — call confirm_render to start");
    let confirm = c.call_tool("confirm_render", serde_json::json!({}));
    assert_eq!(confirm, "render started — poll render_status");

    // 5. poll until done
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut done = false;
    while Instant::now() < deadline {
        let st = c.call_tool("get_render_status", serde_json::json!({}));
        match st["state"].as_str() {
            Some("done") => {
                done = true;
                break;
            }
            Some("failed") => panic!("render failed: {}", st["error"].as_str().unwrap_or("?")),
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    }
    assert!(done, "render did not finish in time");
    assert!(out.exists(), "output missing: {}", out.display());

    let _ = std::fs::remove_dir_all(&dir);
}
