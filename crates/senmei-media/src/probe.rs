use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, specta::Type)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
}

#[derive(Debug, Deserialize)]
struct FfprobeOutput {
    #[serde(default)]
    streams: Vec<Stream>,
    #[serde(default)]
    format: Format,
}

#[derive(Debug, Deserialize)]
struct Stream {
    #[serde(default)]
    codec_type: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    avg_frame_rate: String,
}

#[derive(Debug, Default, Deserialize)]
struct Format {
    #[serde(default)]
    duration: Option<String>,
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(path)
        .output()?;

    if !output.status.success() {
        return Err(Error::Command(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let parsed: FfprobeOutput = serde_json::from_slice(&output.stdout)?;
    let stream = parsed
        .streams
        .iter()
        .find(|s| s.codec_type == "video")
        .ok_or_else(|| Error::Command("no video stream".into()))?;

    let fps = parse_ratio(&stream.avg_frame_rate).unwrap_or(0.0);
    let duration = parsed
        .format
        .duration
        .as_deref()
        .and_then(|d| d.parse::<f64>().ok())
        .unwrap_or(0.0);

    Ok(VideoInfo {
        width: stream.width,
        height: stream.height,
        fps,
        duration,
    })
}

fn parse_ratio(s: &str) -> Option<f64> {
    let (num, den) = s.split_once('/')?;
    let num: f64 = num.trim().parse().ok()?;
    let den: f64 = den.trim().parse().ok()?;
    if den == 0.0 {
        None
    } else {
        Some(num / den)
    }
}
