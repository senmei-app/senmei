use std::ffi::OsStr;
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

pub type ChildHandle = Arc<Mutex<Child>>;

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
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let mut c = Command::new(cmd);
        // Capture PID before fork so the child's getppid() check is correct.
        let parent = unsafe { libc::getpid() };
        unsafe {
            c.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    libc::kill(libc::getpid(), libc::SIGKILL);
                }
                Ok(())
            });
        }
        c
    }
    #[cfg(not(any(windows, target_os = "linux")))]
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

pub fn kill(child: &ChildHandle) {
    let _ = child.lock().unwrap().kill();
}
