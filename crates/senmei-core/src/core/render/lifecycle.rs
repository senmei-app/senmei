//! Render lifecycle: propose → confirm → run on worker thread, status polling,
//! cancel. All gated behind `#[cfg(feature = "render")]`.

use super::{render, validate, RenderConfig, RenderOpts, StepTimingInfo};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Pending (proposed) render — starts only after an explicit confirm.
static PENDING_RENDER: OnceLock<Mutex<Option<RenderConfig>>> = OnceLock::new();

/// Shared status of the active render, updated from the worker thread.
static RENDER_STATUS: OnceLock<Arc<Mutex<RenderStatus>>> = OnceLock::new();

/// Hard cancel flag for the active render (checked between frames).
pub(super) static CANCEL_RENDER: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Render lifecycle status (polled over MCP; no push notifications yet).
#[derive(Debug, Clone, serde::Serialize, schemars::JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RenderStatus {
    /// idle | running | done | failed
    pub state: String,
    pub frames_processed: u64,
    pub total_frames: u64,
    pub error: Option<String>,
    /// Per-step timing once the render finishes (FPS benchmark).
    pub steps: Vec<StepTimingInfo>,
}

impl Default for RenderStatus {
    fn default() -> Self {
        Self {
            state: "idle".into(),
            frames_processed: 0,
            total_frames: 0,
            error: None,
            steps: Vec::new(),
        }
    }
}

/// Propose a render: validates and parks it. Does NOT start — the confirm
/// gate requires `confirm_render` first.
pub fn propose_render(config: RenderConfig) -> Result<String, String> {
    validate(&config)?;
    let slot = PENDING_RENDER.get_or_init(|| Mutex::new(None));
    *slot.lock().unwrap() = Some(config);
    Ok("render proposed — call confirm_render to start".into())
}

/// Starts on a worker thread; poll [`render_status`], abort via [`cancel_render`].
pub fn confirm_render() -> Result<String, String> {
    let slot = PENDING_RENDER.get_or_init(|| Mutex::new(None));
    let config = slot
        .lock()
        .unwrap()
        .take()
        .ok_or_else(|| "no pending render; propose_render first".to_string())?;
    let status = RENDER_STATUS
        .get_or_init(|| Arc::new(Mutex::new(RenderStatus::default())))
        .clone();
    {
        let mut s = status.lock().unwrap();
        if s.state == "running" {
            return Err("a render is already running".into());
        }
        *s = RenderStatus {
            state: "running".into(),
            ..Default::default()
        };
    }
    std::thread::spawn(move || {
        let progress_status = status.clone();
        let result = render(&config, &RenderOpts::default(), move |p| {
            let mut s = progress_status.lock().unwrap();
            s.frames_processed = p.frames_processed;
            s.total_frames = p.total_frames;
        });
        let mut s = status.lock().unwrap();
        match result {
            Ok(steps) => {
                s.state = "done".into();
                s.steps = steps;
            }
            Err(e) => {
                s.state = "failed".into();
                s.error = Some(e);
            }
        }
    });
    Ok("render started — poll render_status".into())
}

pub fn render_status() -> RenderStatus {
    RENDER_STATUS
        .get()
        .map(|s| s.lock().unwrap().clone())
        .unwrap_or_default()
}

/// Sets a flag checked between frames (not instant).
pub fn cancel_render() {
    if let Some(c) = CANCEL_RENDER.get() {
        c.store(true, Ordering::Relaxed);
        log::info!("render cancelled (flag set)");
    }
}
