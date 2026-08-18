//! Persistent decode streams for smooth monitor playback: one long-lived
//! ffmpeg process per file feeds rawvideo frames through a pipe, so playing
//! back reads the next frame instead of spawning a process per frame.

use std::collections::HashMap;
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
    streams: HashMap<PathBuf, PreviewStream>,
}

impl PreviewCache {
    pub fn new(ffmpeg: PathBuf) -> Self {
        Self {
            ffmpeg,
            streams: HashMap::new(),
        }
    }

    /// Return the frame nearest `position_ms` (source time). Reads the next
    /// frame from a warm stream when contiguous; fast-seeks on a jump or EOF.
    pub fn frame(&mut self, input: &str, position_ms: f64) -> Result<Frame> {
        let key = PathBuf::from(input);
        if !self.streams.contains_key(&key) && self.streams.len() >= MAX_STREAMS {
            let victim = self.streams.keys().next().cloned().unwrap();
            self.streams.remove(&victim);
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
            self.streams
                .insert(key.clone(), Self::open(&self.ffmpeg, input, position_ms)?);
        }
        let s = self.streams.get_mut(&key).unwrap();
        let mut reopened = 0;
        loop {
            if s.finished {
                if reopened >= 2 {
                    // Best effort at EOF: hand back the last decoded frame, or
                    // binary-search a valid position when the container
                    // duration is wrong (audio can over-run the video range).
                    if let Some(f) = s.last_frame.clone() {
                        return Ok(f);
                    }
                    if let Some(frame) = self.find_valid_frame(input, position_ms)? {
                        return Ok(frame);
                    }
                    return Err(Error::Command("no frame available at position".into()));
                }
                reopened += 1;
                // Re-seek clamped to the end so near-end requests still get a
                // frame instead of landing past EOF.
                let clamped = (position_ms.min(s.end_ms - s.frame_ms)).max(0.0);
                *s = Self::open(&self.ffmpeg, input, clamped)?;
            }
            // Cheap catch-up: if the request is ahead of the decoded position,
            // skip frames (~1ms pipe read each) instead of re-seeding ffmpeg.
            let mut guard = 0;
            while position_ms > s.next_frame_ms + s.frame_ms && guard < MAX_CATCHUP {
                match s.decoder.next_frame() {
                    Ok(Some(f)) => {
                        s.next_frame_ms += s.frame_ms;
                        s.last_frame = Some(f);
                        guard += 1;
                    }
                    Ok(None) => {
                        s.finished = true;
                        break;
                    }
                    Err(e) => return Err(e),
                }
            }
            if s.finished {
                continue;
            }
            return match s.decoder.next_frame() {
                Ok(Some(f)) => {
                    s.next_frame_ms += s.frame_ms;
                    s.last_frame = Some(f.clone());
                    Ok(f)
                }
                Ok(None) => {
                    s.finished = true;
                    continue;
                }
                Err(e) => Err(e),
            };
        }
    }

    fn open(ffmpeg: &Path, input: &str, position_ms: f64) -> Result<PreviewStream> {
        let decoder = Decoder::open_with_range(
            ffmpeg,
            Path::new(input),
            position_ms.max(0.0) as u64,
            None,
            crate::Tonemap::Auto,
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

    /// Binary-search a position that yields frames. The container duration can
    /// over-report when copied audio runs past a ranged render, so a request
    /// may land beyond the real video end; this finds the last valid position.
    /// Returns the first frame there, or `None` if nothing is decodable.
    fn find_valid_frame(&self, input: &str, position_ms: f64) -> Result<Option<Frame>> {
        let mut lo = 0.0;
        let mut hi = position_ms;
        for _ in 0..8 {
            if hi - lo < 500.0 {
                break;
            }
            let mid = (lo + hi) / 2.0;
            let mut probe = Self::open(&self.ffmpeg, input, mid)?;
            match probe.decoder.next_frame() {
                Ok(Some(_)) => lo = mid,
                _ => hi = mid,
            }
        }
        let mut s = Self::open(&self.ffmpeg, input, lo)?;
        s.decoder.next_frame().map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ranged render copies source audio past the video range unless
    /// `-shortest` is set, so the container duration over-reports and a scrub
    /// beyond the real video end used to hard-error. The binary-search fallback
    /// must return a frame instead.
    #[test]
    #[ignore = "needs a locally rendered mkv with wrong container duration"]
    fn out_of_range_position_still_returns_a_frame() {
        let f = "/home/mzach/Videos/Neo_Ranga_01_test1234_shuffle-cugan_x2_24.mkv";
        if !Path::new(f).exists() {
            return;
        }
        let mut cache = PreviewCache::new("ffmpeg".into());
        let frame = cache
            .frame(f, 100_000.0)
            .expect("frame at out-of-range pos");
        assert!(frame.width > 0 && frame.height > 0);
    }
}
