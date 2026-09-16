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
    let mut pending = slot.lock().unwrap();
    if pending.is_some() {
        return Err("a render is already pending; confirm or cancel it first".into());
    }
    *pending = Some(config);
    Ok("render proposed — call confirm_render to start".into())
}

/// Starts on a worker thread; poll [`render_status`], abort via [`cancel_render`].
pub fn confirm_render() -> Result<String, String> {
    let status = RENDER_STATUS
        .get_or_init(|| Arc::new(Mutex::new(RenderStatus::default())))
        .clone();
    {
        let mut s = status.lock().unwrap();
        if s.state == "running" {
            return Err("a render is already running".into());
        }
        let slot = PENDING_RENDER.get_or_init(|| Mutex::new(None));
        let config = slot
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| "no pending render; propose_render first".to_owned())?;
        *s = RenderStatus {
            state: "running".into(),
            ..Default::default()
        };
        // Reset before spawn so a concurrent cancel_render() is not lost.
        CANCEL_RENDER
            .get_or_init(|| Arc::new(AtomicBool::new(false)))
            .store(false, Ordering::Relaxed);
        drop(s);
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
    }
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

/// Discard a pending (proposed but not yet confirmed) render config.
pub fn discard_pending_render() {
    let slot = PENDING_RENDER.get_or_init(|| Mutex::new(None));
    let mut pending = slot.lock().unwrap();
    if pending.take().is_some() {
        log::info!("pending render discarded");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Tests share the process-wide OnceLock statics; serialize them.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn config() -> RenderConfig {
        RenderConfig {
            input: "input.mp4".into(),
            output: "output.mp4".into(),
            ..Default::default()
        }
    }

    fn reset() {
        *PENDING_RENDER
            .get_or_init(|| Mutex::new(None))
            .lock()
            .unwrap() = None;
        *RENDER_STATUS
            .get_or_init(|| Arc::new(Mutex::new(RenderStatus::default())))
            .lock()
            .unwrap() = RenderStatus::default();
    }

    #[test]
    fn pending_render_is_not_overwritten_or_discarded_while_running() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        propose_render(config()).unwrap();
        assert!(propose_render(config()).is_err());

        RENDER_STATUS
            .get_or_init(|| Arc::new(Mutex::new(RenderStatus::default())))
            .lock()
            .unwrap()
            .state = "running".into();
        assert!(confirm_render().is_err());
        assert!(PENDING_RENDER.get().unwrap().lock().unwrap().is_some());
        reset();
    }

    #[test]
    fn discard_pending_render_clears_proposal() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        reset();
        propose_render(config()).unwrap();
        assert!(PENDING_RENDER.get().unwrap().lock().unwrap().is_some());
        discard_pending_render();
        assert!(PENDING_RENDER.get().unwrap().lock().unwrap().is_none());
        reset();
    }
}
