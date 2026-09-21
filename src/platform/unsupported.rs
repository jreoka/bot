//! Everything else, so the build is a clean refusal rather than a compile error.
//!
//! The model, the reward, the PPO update and `--selftest` all run here; only
//! driving a real window needs a platform backend, and there are three.

use std::collections::HashSet;

use super::{Handle, WindowInfo};

pub const SHELL_TITLES: [&str; 0] = [];

pub fn is_shell_process(_process: &str) -> bool {
    false
}

pub fn is_non_game_process(_process: &str) -> bool {
    false
}

pub fn is_supported() -> bool {
    false
}

pub fn capability_report() -> Vec<String> {
    vec![
        format!(
            "No window backend for this platform ({}). Supported: Windows, Linux/X11, \
             macOS.",
            std::env::consts::OS
        ),
        "The model, the reward, the trainer and --selftest do not need one and work here."
            .to_string(),
    ]
}

pub fn get_process_name(_pid: u32) -> String {
    String::new()
}

pub fn own_process_ids() -> HashSet<u32> {
    let mut ids = HashSet::new();
    ids.insert(std::process::id());
    ids
}

pub fn window_is_own_process(_handle: Handle) -> bool {
    false
}

pub fn enumerate_windows(_min_area: i32) -> Vec<WindowInfo> {
    Vec::new()
}

pub fn window_entry(_handle: Handle) -> Option<WindowInfo> {
    None
}

pub fn console_window() -> Handle {
    0
}

pub fn foreground_window() -> Handle {
    0
}

pub fn client_size(_handle: Handle) -> (i32, i32) {
    (0, 0)
}

pub fn is_minimized(_handle: Handle) -> bool {
    false
}

pub fn is_window(_handle: Handle) -> bool {
    false
}

pub fn window_title(_handle: Handle) -> String {
    String::new()
}

pub fn get_window_pid(_handle: Handle) -> u32 {
    0
}

pub use super::unimplemented::{FrameGrabber, Hotkeys, Injector, KeyRecorder};
