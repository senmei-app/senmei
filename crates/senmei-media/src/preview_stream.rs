//! Persistent decode streams for smooth monitor playback: one long-lived
//! ffmpeg process per file feeds rawvideo frames through a pipe, so playing
//! back reads the next frame instead of spawning a process per frame.

use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use crate::decoder::Decoder;
use crate::frame::Frame;
use crate::{Error, Result};

/// Max streams kept warm (source + result side by side in compare).
const MAX_STREAMS: usize = 2;
/// A request is a jump (re-seek) when it's outside this window around the
/// next decoded frame; within it we read cheaply from the pipe.
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
    ffmpeg: PathBuf,
    /// Preview decode budget: longest edge after downscale (None = full res).
    max_dim: Option<u32>,
    streams: HashMap<PathBuf, PreviewStream>,
    /// LRU access order (back = most recent) for stream eviction.
    order: VecDeque<PathBuf>,
}

impl PreviewCache {
    pub fn new(ffmpeg: PathBuf, max_dim: Option<u32>) -> Self {
        Self {
            ffmpeg,
            max_dim,
            streams: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Return the frame nearest `position_ms` (source time). Reads the next
    /// frame from a warm stream when contiguous; fast-seeks on a jump or EOF.
    pub fn frame(&mut self, input: &str, position_ms: f64) -> Result<Frame> {
        let key = PathBuf::from(input);
        // LRU: evict the least-recently-used stream when at capacity.
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
                Self::open(&self.ffmpeg, input, position_ms, self.max_dim)?,
            );
        }
        self.order.retain(|k| k != &key);
        self.order.push_back(key.clone());

        let s = self.streams.get_mut(&key).unwrap();
        // Cheap catch-up: request ahead of the decoded position within the
        // contiguous window — skip frames instead of re-seeding ffmpeg.
        let mut guard = 0;
        while !s.finished && position_ms > s.next_frame_ms + s.frame_ms && guard < MAX_CATCHUP {
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
        if s.finished {
            // EOF: hand back the last decoded frame (stable at the tail).
            return s
                .last_frame
                .clone()
                .ok_or_else(|| Error::Command("no frame available at position".into()));
        }
        match s.decoder.next_frame() {
            Ok(Some(f)) => {
                s.next_frame_ms += s.frame_ms;
                s.last_frame = Some(f.clone());
                Ok(f)
            }
            Ok(None) => {
                // EOF: return the last decoded frame; if the request landed
                // past the real video end (nothing decoded), re-seek once to
                // the last valid position — the video-stream duration makes
                // `end_ms - frame_ms` a decodable frame.
                if let Some(f) = s.last_frame.clone() {
                    return Ok(f);
                }
                let clamped = (position_ms.min(s.end_ms - s.frame_ms)).max(0.0);
                *s = Self::open(&self.ffmpeg, input, clamped, self.max_dim)?;
                match s.decoder.next_frame() {
                    Ok(Some(f)) => {
                        s.next_frame_ms += s.frame_ms;
                        s.last_frame = Some(f.clone());
                        Ok(f)
                    }
                    _ => Err(Error::Command("no frame available at position".into())),
                }
            }
            Err(e) => Err(e),
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
        let mut cache = PreviewCache::new("ffmpeg".into(), None);
        let frame = cache
            .frame(&f, 100_000.0)
            .expect("frame at out-of-range pos");
        assert!(frame.width > 0 && frame.height > 0);
    }
}
