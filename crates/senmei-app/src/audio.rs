//! Native audio playback for the preview monitor. WebKitGTK can't play media
//! over Tauri's `asset://` scheme (its GStreamer backend doesn't know the
//! scheme), so audio is decoded and played here via rodio, driven by IPC.

use std::cell::RefCell;
use std::io::BufReader;
use std::time::Duration;

struct Player {
    _stream: rodio::OutputStream,
    handle: rodio::OutputStreamHandle,
    sink: Option<rodio::Sink>,
    volume: f32,
}

// rodio's OutputStream is !Send and must live on one thread; Tauri runs sync
// commands on the main thread, so a thread-local keeps it sound.
thread_local! {
    static PLAYER: RefCell<Option<Player>> = RefCell::new(None);
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
            });
        }
        f(guard.as_mut().unwrap())
    })
}

/// Load an extracted audio file (FLAC/lossless PCM); playback stays paused
/// until `audio_play`.
#[tauri::command]
#[specta::specta]
pub fn audio_load(path: String) -> Result<(), String> {
    with_player(|p| {
        p.sink = None;
        let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
        let source = rodio::Decoder::new(BufReader::new(file)).map_err(|e| e.to_string())?;
        let sink = rodio::Sink::try_new(&p.handle).map_err(|e| e.to_string())?;
        sink.pause();
        sink.set_volume(p.volume);
        sink.append(source);
        p.sink = Some(sink);
        Ok(())
    })
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

/// Drop the current audio source so a stale track can't play while the next
/// one is being extracted.
#[tauri::command]
#[specta::specta]
pub fn audio_clear() -> Result<(), String> {
    with_player(|p| {
        p.sink = None;
        Ok(())
    })
}

#[tauri::command]
#[specta::specta]
pub fn audio_seek(position_ms: f64) -> Result<(), String> {
    with_player(|p| {
        if let Some(s) = &p.sink {
            s.try_seek(Duration::from_millis(position_ms.max(0.0) as u64))
                .map_err(|e| e.to_string())?;
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The preview audio path is codec-agnostic: any source is re-encoded to
    /// lossless FLAC by our FFmpeg, and rodio must be able to decode it
    /// (regression for the old MP3/M4A-unsupported targets).
    #[test]
    fn extract_audio_flac_is_rodio_decodable() {
        let ffmpeg = std::env::var("SENMEI_FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
        let dir = std::env::temp_dir();
        let src = dir.join("senmei_audio_src.wav");
        let out = dir.join("senmei_audio_out.flac");
        let ok = std::process::Command::new(&ffmpeg)
            .args([
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:a",
                "pcm_s16le",
            ])
            .arg(&src)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "failed to generate source audio");
        senmei_media::extract_audio(std::path::Path::new(&ffmpeg), &src, &out)
            .expect("extract to flac");
        let file = std::fs::File::open(&out).expect("flac file");
        rodio::Decoder::new(std::io::BufReader::new(file))
            .expect("rodio decodes the flac preview track");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&out);
    }
}
