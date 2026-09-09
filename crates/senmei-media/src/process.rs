use std::ffi::OsStr;
use std::process::Command;

/// A `Command` whose child never pops a console window (Windows: ffmpeg/
/// ffprobe spawns flash otherwise).
pub fn hidden<S: AsRef<OsStr>>(cmd: S) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new(cmd);
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        c
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new(cmd);
        // Kill ffmpeg with senmei — no orphans on Ctrl+C/crash/SIGKILL.
        unsafe {
            c.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                Ok(())
            });
        }
        c
    }
    #[cfg(not(any(windows, unix)))]
    {
        Command::new(cmd)
    }
}

/// Run a command, returning its stdout on success.
pub fn command_output(cmd: &str, args: &[&str]) -> Option<String> {
    hidden(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
}

/// SIGKILL a child by pid (unix); non-unix relies on in-process `Child::kill`.
pub fn kill(pid: u32) {
    #[cfg(unix)]
    // Safety: pid belongs to a live child we spawned; SIGKILL is signal-safe.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = pid;
}
