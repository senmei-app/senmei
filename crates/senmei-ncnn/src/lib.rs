//! Safe wrapper around the NCNN C++ shim (bindgen-generated FFI).

use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl Error {
    fn new(msg: impl Into<String>) -> Self {
        Error(msg.into())
    }
}

/// An NCNN network with its own instance/option; CPU or Vulkan.
pub struct Engine {
    ptr: *mut NcnnEngine,
}

// The shim is callable from any thread; a loaded `ncnn::Net` is read-only and
// each `infer` uses its own Extractor, so sharing an `Engine` is safe.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

impl Engine {
    /// Create an engine; `gpu` requests the Vulkan backend (falls back to CPU
    /// when no Vulkan device is available).
    pub fn new(gpu: bool) -> Result<Engine> {
        let ptr = unsafe { ncnn_engine_new(gpu as i32) };
        if ptr.is_null() {
            return Err(last_error());
        }
        Ok(Engine { ptr })
    }

    pub fn load(&self, param: &Path, bin: &Path) -> Result<()> {
        let param = cstr(param)?;
        let bin = cstr(bin)?;
        let r = unsafe { ncnn_engine_load(self.ptr, param.as_ptr(), bin.as_ptr()) };
        if r != 0 {
            return Err(last_error());
        }
        Ok(())
    }

    /// Run inference on a planar NCHW (1, 3, h, w) float input.
    /// Returns planar NCHW output data + (out_h, out_w).
    pub fn infer(&self, data: &[f32], h: usize, w: usize) -> Result<(Vec<f32>, usize, usize)> {
        let mut out: *mut f32 = ptr::null_mut();
        let mut oh = 0i32;
        let mut ow = 0i32;
        let r = unsafe {
            ncnn_engine_infer(
                self.ptr,
                data.as_ptr(),
                h as i32,
                w as i32,
                &mut out,
                &mut oh,
                &mut ow,
            )
        };
        if r != 0 {
            return Err(last_error());
        }
        let total = 3 * oh as usize * ow as usize;
        let mut vec = Vec::with_capacity(total);
        unsafe {
            ptr::copy_nonoverlapping(out, vec.as_mut_ptr(), total);
            vec.set_len(total);
            ncnn_free(out);
        }
        Ok((vec, oh as usize, ow as usize))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { ncnn_engine_destroy(self.ptr) };
    }
}

fn cstr(path: &Path) -> Result<CString> {
    let s = path
        .as_os_str()
        .to_str()
        .ok_or_else(|| Error::new("path is not UTF-8"))?;
    CString::new(s).map_err(|_| Error::new("path contains NUL"))
}

fn last_error() -> Error {
    let msg = unsafe { CStr::from_ptr(ncnn_engine_last_error()) }
        .to_string_lossy()
        .into_owned();
    Error::new(msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn new_and_destroy() {
        let engine = Engine::new(false).unwrap();
        drop(engine);
    }

    #[test]
    #[ignore = "requires models/up2x-no-denoise.{param,bin} (from realcugan-ncnn-vulkan)"]
    fn loads_model_and_upscales_2x() {
        let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models"));
        let param = dir.join("up2x-no-denoise.param");
        let bin = dir.join("up2x-no-denoise.bin");
        if !param.exists() || !bin.exists() {
            eprintln!("model files missing, skipping");
            return;
        }
        let engine = Engine::new(false).unwrap();
        engine.load(&param, &bin).unwrap();
        // Real-CUGAN upcunet crops a fixed border: out = 2*h - 72.
        for (in_size, out_size) in [(64usize, 56usize), (96, 120), (128, 184), (256, 440)] {
            let data = vec![0.5f32; 3 * in_size * in_size];
            let (out, h, w) = engine.infer(&data, in_size, in_size).unwrap();
            assert_eq!((h, w), (out_size, out_size), "input {in_size}x{in_size}");
            assert_eq!(out.len(), 3 * out_size * out_size);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
