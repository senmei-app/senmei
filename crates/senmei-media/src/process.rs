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
    #[cfg(not(windows))]
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
