//! Preview audio via rodio: WebKitGTK can't play Tauri's `asset://` scheme,
//! so ffmpeg streams PCM here (no re-encode, no rodio-codec dep).

use std::cell::RefCell;
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use rodio::Source;

struct PcmSource {
    rx: mpsc::Receiver<Vec<u8>>,
    buf: std::vec::IntoIter<i16>,
    sample_rate: u32,
    channels: u16,
}

impl Iterator for PcmSource {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        loop {
            if let Some(v) = self.buf.next() {
                return Some(v as f32 / 32768.0);
            }
            match self.rx.recv() {
                Ok(chunk) => {
                    let samples: Vec<i16> = chunk
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    self.buf = samples.into_iter();
                }
                Err(_) => return None, // pipe closed = end of stream
            }
        }
    }
}

impl Source for PcmSource {
    fn current_frame_len(&self) -> Option<usize> {
        None
    }
    fn channels(&self) -> u16 {
        self.channels
    }
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

struct Player {
    _stream: rodio::OutputStream,
    handle: rodio::OutputStreamHandle,
    sink: Option<rodio::Sink>,
    volume: f32,
    input: Option<String>,
    pipe: Option<senmei_media::PcmPipe>,
}

// OutputStream is !Send; a thread-local keeps it sound (commands on main thread).
thread_local! {
    static PLAYER: RefCell<Option<Player>> = const { RefCell::new(None) };
}

fn with_player<T>(f: impl FnOnce(&mut Player) -> Result<T, String>) -> Result<T, String> {
    PLAYER.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            let (stream, handle) = rodio::OutputStream::try_default().map_err(|e| e.to_string())?;
            *guard = Some(Player {
                _stream: stream,
                handle,
                sink: None,
                volume: 1.0,
                input: None,
                pipe: None,
            });
        }
        f(guard.as_mut().unwrap())
    })
}

/// Pre-roll ~200 ms of PCM so rodio doesn't starve during ffmpeg's seek decode.
fn preroll_pcm(rx: &mpsc::Receiver<Vec<u8>>) -> Vec<i16> {
    const PREROLL_SAMPLES: usize = 48_000 * 2 / 5; // 200 ms of stereo
    let deadline = Instant::now() + Duration::from_millis(150);
    let mut out = Vec::new();
    while out.len() < PREROLL_SAMPLES {
        let remain = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remain) {
            Ok(chunk) => out.extend(
                chunk
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]])),
            ),
            Err(_) => break,
        }
    }
    out
}

fn start(p: &mut Player, input: &str, position_ms: f64, playing: bool) -> Result<(), String> {
    if let Some(mut pipe) = p.pipe.take() {
        pipe.stop();
    }
    p.sink = None;
    let ffmpeg = senmei_media::resolve(&crate::store::data_dir());
    let (pipe, rx) = senmei_media::stream_pcm(&ffmpeg, Path::new(input), position_ms, 48_000)
        .map_err(|e| e.to_string())?;
    let preroll = preroll_pcm(&rx);
    let source = PcmSource {
        rx,
        buf: preroll.into_iter(),
        sample_rate: 48_000,
        channels: 2,
    };
    let sink = rodio::Sink::try_new(&p.handle).map_err(|e| e.to_string())?;
    sink.pause();
    sink.set_volume(p.volume);
    sink.append(source);
    if playing {
        sink.play();
    }
    p.sink = Some(sink);
    p.pipe = Some(pipe);
    p.input = Some(input.to_string());
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn audio_load(input: String, position_ms: f64) -> Result<(), String> {
    with_player(|p| start(p, &input, position_ms, false))
}

#[tauri::command]
#[specta::specta]
pub fn audio_play() -> Result<(), String> {
    with_player(|p| {
        if let Some(s) = &p.sink {
            s.play();
        }
        Ok(())
    })
}

#[tauri::command]
#[specta::specta]
pub fn audio_pause() -> Result<(), String> {
    with_player(|p| {
        if let Some(s) = &p.sink {
            s.pause();
        }
        Ok(())
    })
}

#[tauri::command]
#[specta::specta]
pub fn audio_clear() -> Result<(), String> {
    with_player(|p| {
        if let Some(mut pipe) = p.pipe.take() {
            pipe.stop();
        }
        p.sink = None;
        p.input = None;
        Ok(())
    })
}

/// Seek = restart the pipe at position (keeps play state).
#[tauri::command]
#[specta::specta]
pub fn audio_seek(position_ms: f64) -> Result<(), String> {
    with_player(|p| {
        let input = p
            .input
            .clone()
            .ok_or_else(|| "no audio loaded".to_string())?;
        let playing = p.sink.as_ref().map(|s| !s.is_paused()).unwrap_or(false);
        start(p, &input, position_ms, playing)
    })
}

#[tauri::command]
#[specta::specta]
pub fn audio_set_volume(volume: f64) -> Result<(), String> {
    with_player(|p| {
        p.volume = volume.clamp(0.0, 1.0) as f32;
        if let Some(s) = &p.sink {
            s.set_volume(p.volume);
        }
        Ok(())
    })
}
