use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, specta::Type)]
#[serde(rename_all = "camelCase")]
pub struct VideoInfo {
    /// Display width after applying rotation (stored dims for unrotated video).
    pub width: u32,
    /// Display height after applying rotation.
    pub height: u32,
    pub fps: f64,
    pub duration: f64,
    /// Accurate video-stream duration (container over-reports with copied
    /// audio); internal decode cap, not serialized.
    #[serde(skip)]
    pub video_duration: f64,
    /// Clockwise rotation needed for display, normalized to 0/90/180/270.
    pub rotation: u32,
    /// Source color transfer characteristic (e.g. "smpte2084" = PQ).
    pub color_transfer: Option<String>,
    /// Source color primaries (e.g. "bt2020").
    pub color_primaries: Option<String>,
    /// Video codec name (e.g. "h264", "hevc", "av1").
    pub video_codec: Option<String>,
    /// First audio stream's codec name (e.g. "aac", "opus").
    pub audio_codec: Option<String>,
    /// Video pixel format (e.g. "yuv420p").
    pub pix_fmt: Option<String>,
}

impl VideoInfo {
    /// HDR if the source uses a PQ/HLG/DCI transfer (tonemapping to SDR applies).
    pub fn is_hdr(&self) -> bool {
        matches!(
            self.color_transfer.as_deref(),
            Some("smpte2084" | "arib-std-b67" | "smpte428-1")
        )
    }
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
    codec_name: String,
    #[serde(default)]
    width: u32,
    #[serde(default)]
    height: u32,
    #[serde(default)]
    avg_frame_rate: String,
    #[serde(default)]
    duration: Option<String>,
    #[serde(default)]
    color_transfer: Option<String>,
    #[serde(default)]
    color_primaries: Option<String>,
    #[serde(default)]
    pix_fmt: Option<String>,
    #[serde(default)]
    tags: HashMap<String, String>,
    #[serde(default)]
    side_data_list: Vec<SideData>,
}

#[derive(Debug, Default, Deserialize)]
struct SideData {
    #[serde(default)]
    rotation: Option<i32>,
}

#[derive(Debug, Default, Deserialize)]
struct Format {
    #[serde(default)]
    duration: Option<String>,
}

/// Clockwise display rotation in degrees, normalized to 0/90/180/270.
/// ffprobe reports it via the DisplayMatrix side data (`side_data_list.rotation`)
/// or a rotation stream tag (case-insensitive: MP4 `rotate`, MKV `ROTATE`).
fn stream_rotation(stream: &Stream) -> u32 {
    let rot = stream
        .side_data_list
        .iter()
        .find_map(|s| s.rotation)
        .or_else(|| {
            stream
                .tags
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("rotate"))
                .and_then(|(_, v)| v.parse::<i32>().ok())
        })
        .unwrap_or(0);
    let norm = ((rot % 360) + 360) % 360;
    match norm {
        0 | 90 | 180 | 270 => norm as u32,
        _ => 0,
    }
}

pub fn probe(ffprobe: &Path, path: &Path) -> Result<VideoInfo> {
    let output = Command::new(ffprobe)
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
    parse(parsed)
}

fn parse(parsed: FfprobeOutput) -> Result<VideoInfo> {
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
    // Video-stream duration beats the container duration when they disagree
    // (copied audio can over-report the container).
    let video_duration = stream
        .duration
        .as_deref()
        .and_then(|d| d.parse::<f64>().ok())
        .filter(|d| *d > 0.0)
        .unwrap_or(duration);

    // Display dimensions after rotation: 90°/270° swap stored w/h.
    let rotation = stream_rotation(stream);
    let (width, height) = if rotation == 90 || rotation == 270 {
        (stream.height, stream.width)
    } else {
        (stream.width, stream.height)
    };

    let audio_codec = parsed
        .streams
        .iter()
        .find(|s| s.codec_type == "audio")
        .map(|s| s.codec_name.clone())
        .filter(|c| !c.is_empty());

    Ok(VideoInfo {
        width,
        height,
        fps,
        duration,
        video_duration,
        rotation,
        color_transfer: stream.color_transfer.clone(),
        color_primaries: stream.color_primaries.clone(),
        video_codec: Some(stream.codec_name.clone()).filter(|c| !c.is_empty()),
        audio_codec,
        pix_fmt: stream.pix_fmt.clone(),
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

#[cfg(test)]
mod tests {
    use super::{parse, FfprobeOutput, VideoInfo};

    fn info(transfer: Option<&str>) -> VideoInfo {
        VideoInfo {
            width: 1920,
            height: 1080,
            fps: 24.0,
            duration: 1.0,
            video_duration: 1.0,
            rotation: 0,
            color_transfer: transfer.map(String::from),
            color_primaries: Some("bt2020".into()),
            video_codec: Some("h264".into()),
            audio_codec: Some("aac".into()),
            pix_fmt: Some("yuv420p".into()),
        }
    }

    #[test]
    fn hdr_detection() {
        assert!(info(Some("smpte2084")).is_hdr(), "PQ");
        assert!(info(Some("arib-std-b67")).is_hdr(), "HLG");
        assert!(info(Some("smpte428-1")).is_hdr(), "DCI-P3 transfer");
        assert!(!info(Some("bt709")).is_hdr(), "SDR");
        assert!(!info(None).is_hdr(), "no transfer");
    }

    #[test]
    fn codec_parsing() {
        let parsed: FfprobeOutput = serde_json::from_str(
            r#"{
                "streams": [
                    {"codec_type":"video","codec_name":"hevc","width":1280,"height":720,
                     "avg_frame_rate":"24000/1001","pix_fmt":"yuv420p10le"},
                    {"codec_type":"audio","codec_name":"opus"}
                ],
                "format": {"duration":"12.5"}
            }"#,
        )
        .unwrap();
        let v = parse(parsed).unwrap();
        assert_eq!(v.video_codec.as_deref(), Some("hevc"));
        assert_eq!(v.audio_codec.as_deref(), Some("opus"));
        assert_eq!(v.pix_fmt.as_deref(), Some("yuv420p10le"));
        assert_eq!((v.width, v.height), (1280, 720));
    }

    #[test]
    fn codec_parsing_without_audio() {
        let parsed: FfprobeOutput = serde_json::from_str(
            r#"{
                "streams": [
                    {"codec_type":"video","codec_name":"av1","width":3840,"height":2160,
                     "avg_frame_rate":"60/1"}
                ],
                "format": {"duration":"1"}
            }"#,
        )
        .unwrap();
        let v = parse(parsed).unwrap();
        assert_eq!(v.video_codec.as_deref(), Some("av1"));
        assert_eq!(v.audio_codec, None);
        assert_eq!(v.pix_fmt, None);
    }
}
