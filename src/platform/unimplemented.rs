//! What a platform that has no window layer yet hands back.
//!
//! The point of these types is that *nothing happens*: a run on a platform
//! without a backend captures nothing, sends nothing, and says why. A stub that
//! silently returned an empty frame would look like a frozen game, and a stub
//! that silently dropped input would look like a policy that had stopped
//! pressing keys - both of which the bot would report as a health problem in the
//! game rather than as a missing backend.

use crate::capture::Frame;
use crate::config::FocusPolicy;
use crate::keys::Recording;
use crate::platform::{Handle, HotkeyAction, WindowInfo};

/// Why this platform cannot drive a window, in one line.
fn reason() -> String {
    crate::platform::capability_report()
        .into_iter()
        .next()
        .unwrap_or_else(|| "no window backend for this platform".to_string())
}

pub struct FrameGrabber {
    pub last_error: Option<String>,
    pub frames: u64,
    announced: bool,
}

impl FrameGrabber {
    pub fn new(_handle: crate::platform::Handle) -> Self {
        Self {
            last_error: Some(reason()),
            frames: 0,
            announced: false,
        }
    }

    pub fn grab(&mut self) -> Option<Frame> {
        self.frames += 1;
        if !self.announced {
            self.announced = true;
            println!("[Capture] {}", self.last_error.clone().unwrap_or_default());
        }
        None
    }

    pub fn mean_ms(&self) -> f64 {
        0.0
    }
}

pub struct Injector {
    pub policy: FocusPolicy,
    pub suspended: bool,
    pub enabled: bool,
    pub blocked: u64,
    pub failures: u64,
    pub skipped_unfocused: u64,
    pub events_sent: u64,
    pub keys_sent: u64,
    pub mouse_sent: u64,
    pub last_error: Option<String>,
    announced: bool,
}

impl Injector {
    pub fn new(_handle: crate::platform::Handle, policy: FocusPolicy) -> Self {
        Self {
            policy,
            suspended: false,
            enabled: true,
            blocked: 0,
            failures: 0,
            skipped_unfocused: 0,
            events_sent: 0,
            keys_sent: 0,
            mouse_sent: 0,
            last_error: Some(reason()),
            announced: false,
        }
    }

    fn announce(&mut self) {
        if !self.announced {
            self.announced = true;
            self.failures += 1;
            println!(
                "[Input] Nothing is being sent: {}",
                self.last_error.clone().unwrap_or_default()
            );
        }
    }

    pub fn focused(&self) -> bool {
        false
    }

    pub fn acquire_focus(&mut self, _force: bool) -> bool {
        self.announce();
        false
    }

    pub fn begin_action(&mut self) {
        self.announce();
    }

    pub fn end_action(&mut self) {}

    pub fn status_line(&self) -> String {
        format!(
            "[Input] no window layer on this platform - {} event(s) delivered",
            self.events_sent
        )
    }

    pub fn press_vk(&mut self, vk: u32) {
        if vk != 0 {
            self.announce();
        }
    }

    pub fn release_vk(&mut self, vk: u32) {
        if vk != 0 {
            self.announce();
        }
    }

    pub fn tap_vk(&mut self, vk: u32, _seconds: f32) {
        if vk != 0 {
            self.announce();
        }
    }

    pub fn mouse_move(&mut self, _dx: i32, _dy: i32) {
        self.announce();
    }

    pub fn mouse_down(&mut self, _button: &str) {
        self.announce();
    }

    pub fn mouse_up(&mut self, _button: &str) {
        self.announce();
    }

    pub fn click(&mut self, _button: &str, _seconds: f32) {
        self.announce();
    }

    pub fn release_all(&mut self, _held_vks: &[u32]) {}
}

/// Global hotkeys: none, and it says so rather than pretending they work.
pub struct Hotkeys {
    messages: Vec<String>,
}

impl Hotkeys {
    pub fn new(_pause: &str, _save: &str, _quit: &str) -> Self {
        Self {
            messages: vec![reason()],
        }
    }

    pub fn start(&mut self) -> bool {
        false
    }

    pub fn available(&self) -> bool {
        false
    }

    pub fn poll(&mut self) -> Vec<HotkeyAction> {
        Vec::new()
    }

    pub fn drain_messages(&mut self) -> Vec<String> {
        std::mem::take(&mut self.messages)
    }

    pub fn describe(&self) -> String {
        "none".to_string()
    }

    pub fn stop(&mut self) {}
}

/// The calibration recorder: installed, but it cannot see anything.
pub struct KeyRecorder {
    announced: bool,
}

impl KeyRecorder {
    pub fn start(_toggle_vk: u32, _target: Handle, _only_when_focused: bool) -> Self {
        Self { announced: false }
    }

    /// Whether the hooks really went in. On a platform with no backend: no.
    pub fn installed(&mut self) -> bool {
        if !self.announced {
            self.announced = true;
            println!("[Calibrate] {}", reason());
        }
        false
    }

    pub fn pump(&mut self) {}

    pub fn recording(&self) -> bool {
        false
    }

    pub fn toggles(&self) -> u64 {
        0
    }

    pub fn events(&self) -> u64 {
        0
    }

    pub fn stop(&mut self) {}

    pub fn snapshot(&self, _target: Option<&WindowInfo>) -> Recording {
        Recording::default()
    }
}
