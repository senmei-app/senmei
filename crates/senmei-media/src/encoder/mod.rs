mod select;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::frame::Frame;
use crate::{Error, Result};

use select::{
    extract_audio_range, hw_verifier, kvazaar_compat_args, override_codec_args, pick_from_caps,
    set_vaapi_prefer_igpu, vaapi_compat_args, vaapi_device, EncoderPref,
};
#[cfg(test)]
use select::{test_encode, HW_ENCODERS};

pub struct Encoder {
    child: Child,
    stdin: Option<ChildStdin>,
    stderr: Arc<Mutex<String>>,
    stderr_thread: Option<JoinHandle<()>>,
    temp_audio: Option<PathBuf>,
}

/// Fixed inputs for one encode; the per-call ffmpeg `extra_args` ride along
/// separately in [`Encoder::open`].
#[derive(Clone, Copy)]
pub struct EncodeOptions<'a> {
    pub ffmpeg: &'a Path,
    pub input: &'a Path,
    pub output: &'a Path,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub start_ms: u64,
    pub duration_ms: Option<u64>,
}

impl Encoder {
    pub fn open(cfg: &EncodeOptions, extra_args: &[String]) -> Result<Self> {
        let EncodeOptions {
            ffmpeg,
            input,
            output: path,
            width,
            height,
            fps,
            start_ms,
            duration_ms,
        } = *cfg;
        let caps = crate::ffmpeg::probe(ffmpeg).encoders;
        let verify = hw_verifier(ffmpeg);
        let mut extra_args = extra_args.to_vec();
        let mut pref = EncoderPref::Auto;
        if let Some(pos) = extra_args.iter().position(|a| a == "-senmei_encoder") {
            if let Some(v) = extra_args.get(pos + 1) {
                pref = match v.as_str() {
                    "hw" => EncoderPref::Hardware,
                    "sw" => EncoderPref::Software,
                    _ => EncoderPref::Auto,
                };
            }
            extra_args.drain(pos..pos + 2);
        }
        if let Some(pos) = extra_args.iter().position(|a| a == "-senmei_vaapi") {
            let igpu = extra_args
                .get(pos + 1)
                .map(|v| v == "igpu")
                .unwrap_or(false);
            set_vaapi_prefer_igpu(igpu);
            extra_args.drain(pos..pos + 2);
        }
        let vaapi_10bit = extra_args
            .windows(2)
            .any(|w| w[0] == "-pix_fmt" && w[1].starts_with("yuv4") && w[1].contains("10le"));
        let verify_full = |codec: &str| select::test_encode(ffmpeg, codec, width, height);
        let (mut video_codec, mut codec_args) =
            pick_from_caps(&caps, width, height, pref, &verify, &verify_full);
        if let Some(pos) = extra_args.windows(2).position(|w| w[0] == "-c:v") {
            let codec = extra_args[pos + 1].clone();
            extra_args.drain(pos..pos + 2);
            if caps.contains(&codec) {
                video_codec = codec.clone();
                codec_args = override_codec_args(&codec, &extra_args, width, height);
            } else {
                log::warn!("encoder `{codec}` unavailable; falling back to `{video_codec}`");
            }
        }
        if video_codec == "libkvazaar" {
            extra_args = kvazaar_compat_args(&extra_args);
        }
        if video_codec.ends_with("_vaapi") {
            extra_args = vaapi_compat_args(&extra_args);
        }
        let vaapi = video_codec.ends_with("_vaapi").then(vaapi_device).flatten();
        if vaapi.is_some() && !extra_args.iter().any(|a| a == "-qp" || a == "-rc_mode") {
            codec_args = vec!["-qp".into(), "20".into()];
        }
        log::info!(
            "encode {}@{}x{} device={}",
            video_codec,
            width,
            height,
            vaapi
                .as_ref()
                .map(|d| d.display().to_string())
                .unwrap_or_else(|| "cpu".into())
        );
        let mut cmd = crate::process::hidden(ffmpeg);
        cmd.arg("-y");
        if let Some(dev) = &vaapi {
            cmd.args(["-init_hw_device", &format!("vaapi=va:{}", dev.display())]);
            cmd.args(["-filter_hw_device", "va"]);
        }
        cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24"])
            .args(["-s", &format!("{width}x{height}")])
            .args(["-r", &format!("{fps}")])
            .args(["-i", "-"]);
        let want_audio = !extra_args.iter().any(|a| a == "-an");
        let mut temp_audio: Option<PathBuf> = None;
        if want_audio && (start_ms > 0 || duration_ms.is_some()) {
            temp_audio = extract_audio_range(ffmpeg, input, start_ms, duration_ms);
        }
        if let Some(tmp) = &temp_audio {
            cmd.arg("-i").arg(tmp);
        } else {
            if start_ms > 0 {
                cmd.args(["-ss", &format!("{:.3}", start_ms as f64 / 1000.0)]);
            }
            if let Some(dur) = duration_ms {
                cmd.args(["-t", &format!("{:.3}", dur as f64 / 1000.0)]);
            }
            cmd.arg("-i").arg(input);
        }
        cmd.arg("-copyts")
            .args(["-map", "0:v:0", "-map", "1:a:0?"])
            .args(["-shortest"])
            .args(if temp_audio.is_some() {
                vec!["-c:a".to_string(), "copy".to_string()]
            } else {
                Vec::new()
            })
            .args(["-c:v", &video_codec])
            .args(codec_args)
            .args(if vaapi.is_some() {
                let fmt = if vaapi_10bit { "p010" } else { "nv12" };
                ["-vf".to_string(), format!("format={fmt},hwupload")]
            } else {
                ["-pix_fmt".to_string(), "yuv420p".to_string()]
            })
            .args(&extra_args)
            .arg(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn()?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Command("failed to capture ffmpeg stdin".into()))?;
        let stderr_buf = Arc::new(Mutex::new(String::new()));
        let stderr_thread = child.stderr.take().map(|mut e| {
            let buf = stderr_buf.clone();
            std::thread::spawn(move || {
                use std::io::Read;
                let mut s = String::new();
                let _ = e.read_to_string(&mut s);
                *buf.lock().unwrap() = s;
            })
        });

        Ok(Self {
            child,
            stdin: Some(stdin),
            stderr: stderr_buf,
            stderr_thread,
            temp_audio,
        })
    }

    pub fn write_frame(&mut self, frame: &Frame) -> Result<()> {
        if let Some(stdin) = self.stdin.as_mut() {
            if let Err(e) = stdin.write_all(&frame.data) {
                let _ = self.child.kill();
                let _ = self.child.wait();
                let stderr = self.read_stderr();
                return Err(Error::Command(if stderr.is_empty() {
                    format!("ffmpeg encode write failed: {e}")
                } else {
                    format!("ffmpeg encode write failed: {e}\n{stderr}")
                }));
            }
        }
        Ok(())
    }

    fn read_stderr(&mut self) -> String {
        if let Some(h) = self.stderr_thread.take() {
            let _ = h.join();
        }
        let out = self.stderr.lock().unwrap().clone();
        const TAIL: usize = 12;
        let lines: Vec<&str> = out.lines().collect();
        let tail = if lines.len() > TAIL {
            &lines[lines.len() - TAIL..]
        } else {
            &lines[..]
        };
        tail.join("\n").trim().to_string()
    }

    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        let stderr = self.read_stderr();
        log::debug!("ffmpeg encode finished: {status}; stderr tail: {stderr}");
        if status.success() {
            Ok(())
        } else {
            Err(Error::Command(if stderr.is_empty() {
                format!("ffmpeg encode exited with {status}")
            } else {
                format!("ffmpeg encode exited with {status}:\n{stderr}")
            }))
        }
    }

    pub fn abort(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(tmp) = self.temp_audio.take() {
            let _ = std::fs::remove_file(tmp);
        }
    }
}

#[cfg(test)]
mod tests;
