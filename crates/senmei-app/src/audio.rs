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

/// Load an extracted audio file (MP3); playback stays paused until `audio_play`.
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
