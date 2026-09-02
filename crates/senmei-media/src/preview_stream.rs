//! Warm decode streams: one long-lived ffmpeg per file, so playback reads the
//! next frame from the pipe instead of spawning a process per frame.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use crate::decoder::Decoder;
use crate::frame::Frame;
use crate::{Error, Result};

/// Preview decode budget (longest edge) — display-sized, keeps the frame
/// transfer cheap. Only ever downscales.
pub const PREVIEW_MAX_DIM: u32 = 1280;

/// Max streams kept warm (source + result side by side in compare).
const MAX_STREAMS: usize = 2;
/// Re-seek when a request falls outside this window around the next frame.
const CONTIG_TOL_MS: f64 = 300.0;
/// Cap on cheap catch-up frame skips per read (avoids runaway decode).
const MAX_CATCHUP: usize = 1500;

struct PreviewStream {
    decoder: Decoder,
    /// Approx timestamp of the frame the next `next_frame()` returns.
    next_frame_ms: f64,
    frame_ms: f64,
    end_ms: f64,
    /// Last successfully decoded frame, handed back at EOF.
    last_frame: Option<Frame>,
    finished: bool,
}

/// Warm decode streams keyed by input file.
pub struct PreviewCache {
    /// Data dir — ffmpeg is re-resolved per open (a freshly downloaded
    /// portable FFmpeg is picked up without restarting the app).
    data_dir: PathBuf,
    /// Preview decode budget: longest edge after downscale (None = full res).
    max_dim: Option<u32>,
    streams: HashMap<PathBuf, PreviewStream>,
    /// LRU access order (back = most recent) for stream eviction.
    order: VecDeque<PathBuf>,
}

impl PreviewCache {
    pub fn new(data_dir: PathBuf, max_dim: Option<u32>) -> Self {
        Self {
            data_dir,
            max_dim,
            streams: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Return the frame nearest `position_ms` (source time). Reads the next
    /// frame from a warm stream when contiguous; fast-seeks on a jump or EOF.
    pub fn frame(&mut self, input: &str, position_ms: f64) -> Result<Frame> {
        let key = PathBuf::from(input);
        if !self.streams.contains_key(&key) && self.streams.len() >= MAX_STREAMS {
            if let Some(victim) = self.order.pop_front() {
                self.streams.remove(&victim);
            }
        }
        let need_seek = match self.streams.get(&key) {
            None => true,
            Some(s) => {
                s.finished
                    || position_ms < s.next_frame_ms - CONTIG_TOL_MS
                    || position_ms > s.next_frame_ms + CONTIG_TOL_MS
            }
        };
        if need_seek {
            self.streams.insert(
                key.clone(),
                Self::open(
                    &crate::resolve(&self.data_dir),
                    input,
                    position_ms,
                    self.max_dim,
                )?,
            );
        }
        self.order.retain(|k| k != &key);
        self.order.push_back(key.clone());

        let s = self.streams.get_mut(&key).unwrap();
        // Return the nearest frame; never run ahead — that made playback
        // oscillate between re-seek and ahead-read when decode lagged.
        let mut guard = 0;
        while !s.finished
            && position_ms + s.frame_ms / 2.0 >= s.next_frame_ms
            && guard < MAX_CATCHUP
        {
            match s.decoder.next_frame() {
                Ok(Some(f)) => {
                    s.next_frame_ms += s.frame_ms;
                    s.last_frame = Some(f);
                    guard += 1;
                }
                Ok(None) => {
                    s.finished = true;
                }
                Err(e) => return Err(e),
            }
        }
        if let Some(f) = s.last_frame.clone() {
            return Ok(f);
        }
        // On EOF (request past the real video end) re-seek once to a valid
        // position instead of hard-erroring.
        if s.finished {
            let clamped = (position_ms.min(s.end_ms - s.frame_ms)).max(0.0);
            *s = Self::open(
                &crate::resolve(&self.data_dir),
                input,
                clamped,
                self.max_dim,
            )?;
            match s.decoder.next_frame() {
                Ok(Some(f)) => {
                    s.next_frame_ms += s.frame_ms;
                    s.last_frame = Some(f.clone());
                    Ok(f)
                }
                _ => Err(Error::Command("no frame available at position".into())),
            }
        } else {
            match s.decoder.next_frame() {
                Ok(Some(f)) => {
                    s.next_frame_ms += s.frame_ms;
                    s.last_frame = Some(f.clone());
                    Ok(f)
                }
                _ => Err(Error::Command("no frame available at position".into())),
            }
        }
    }

    fn open(
        ffmpeg: &Path,
        input: &str,
        position_ms: f64,
        max_dim: Option<u32>,
    ) -> Result<PreviewStream> {
        let decoder = Decoder::open_with_range(
            ffmpeg,
            Path::new(input),
            position_ms.max(0.0) as u64,
            None,
            crate::Tonemap::Auto,
            max_dim,
        )?;
        let frame_ms = 1000.0 / decoder.fps;
        let end_ms = (decoder.total_frames.max(1) as f64) / decoder.fps * 1000.0;
        Ok(PreviewStream {
            decoder,
            next_frame_ms: position_ms,
            frame_ms,
            end_ms,
            last_frame: None,
            finished: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ranged render copies source audio past the video range unless
    /// `-shortest` is set, so the container duration over-reports and a scrub
    /// beyond the real video end used to hard-error. The re-seek to the last
    /// valid video position must return a frame instead.
    #[test]
    #[ignore = "needs a locally rendered mkv with wrong container duration"]
    fn out_of_range_position_still_returns_a_frame() {
        let Some(f) = std::env::var("SENMEI_TEST_MKV").ok() else {
            return;
        };
        if !Path::new(&f).exists() {
            return;
        }
        let mut cache = PreviewCache::new(std::env::temp_dir(), None);
        let frame = cache
            .frame(&f, 100_000.0)
            .expect("frame at out-of-range pos");
        assert!(frame.width > 0 && frame.height > 0);
    }

    /// Regression: playback frames must never jump backward in time. The old
    /// catch-up read ahead of the request (notably with slow upscaled decodes),
    /// then re-seeked once the request lagged >300ms — alternating between
    /// ahead/behind positions that looked like the image "jumping". Requests
    /// that advance slower than the frame rate must still yield monotonic
    /// frames.
    #[test]
    fn forward_playback_stays_monotonic() {
        let dir = std::env::temp_dir().join("senmei_preview_mono_test");
        std::fs::create_dir_all(&dir).unwrap();
        let video = dir.join("mono.mp4");
        if !video.exists() {
            let ok = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=320x180:r=24:d=4",
                    "-vf",
                    "geq=lum='min(255,N*4)':cb=128:cr=128",
                    "-c:v",
                    "mpeg4",
                    "-pix_fmt",
                    "yuv420p",
                ])
                .arg(&video)
                .status()
                .is_ok_and(|s| s.success());
            if !ok {
                return; // no ffmpeg — skip
            }
        }
        let mut cache = PreviewCache::new(std::env::temp_dir(), None);
        // Requests advance ~1/5 of a frame per call (a 60Hz timer on 24fps
        // video); the decode used to run far ahead and oscillate.
        let mut last_avg = -1.0;
        let mut t = 0.0;
        while t < 2000.0 {
            let frame = cache
                .frame(&video.to_string_lossy(), t)
                .expect("frame during playback");
            // geq sets a uniform luma per frame → avg RGB == the frame's luma.
            let avg = frame.data.iter().map(|&b| b as f64).sum::<f64>() / frame.data.len() as f64;
            assert!(
                avg + 0.5 >= last_avg,
                "frame at {t}ms jumped backward in time (avg {avg:.1} < last {last_avg:.1})"
            );
            last_avg = avg;
            t += 8.0;
        }
    }
}
