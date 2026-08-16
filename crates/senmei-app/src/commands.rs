use std::path::PathBuf;

use serde::Serialize;
use tauri::ipc::Channel;

#[tauri::command]
pub fn health_check() -> String {
    "ok".to_string()
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderProgress {
    pub frames_processed: u64,
    pub total_frames: u64,
}

#[tauri::command]
pub async fn render(
    input: String,
    output: String,
    on_progress: Channel<RenderProgress>,
) -> Result<String, String> {
    let input = PathBuf::from(input);
    let output = PathBuf::from(output);

    tauri::async_runtime::spawn_blocking(move || -> Result<String, String> {
        let steps: Vec<Box<dyn senmei_pipeline::Step>> =
            vec![Box::new(senmei_pipeline::Passthrough)];
        let mut pipeline = senmei_pipeline::Pipeline::new(steps);

        pipeline
            .run(&input, &output, |p| {
                let _ = on_progress.send(RenderProgress {
                    frames_processed: p.frames_processed,
                    total_frames: p.total_frames,
                });
            })
            .map(|_| "ok".to_string())
            .map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| e.to_string())?
}
