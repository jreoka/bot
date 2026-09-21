//! The macOS mechanism behind [`crate::platform`].
//!
//! Window discovery, capture and input injection through CoreGraphics. What is
//! here is the same shape as the Win32 and X11 backends: the same function names,
//! the same policy above them.
//!
//! **This backend has never been run.** There is no Mac in this session, so it is
//! verified by the compiler against the real `core-graphics` and
//! `core-foundation` bindings and by unit tests of its pure parts (the keycode
//! table, the process tables, the window filters). Everything after "it compiles"
//! is unverified, and the first person to run it on a Mac should start with
//! `--platform`, `--list-windows` and `--check-capture`.
//!
//! ### Two permissions, and neither can be asked for in code
//!
//! * **Screen Recording** for capture. Without it `CGWindowListCreateImage`
//!   returns the desktop wallpaper and the menu bar rather than the window - a
//!   picture that looks plausible, changes when the desktop changes, and contains
//!   none of the game. `CGPreflightScreenCaptureAccess` is checked here and
//!   reported, because the alternative is a bot that trains on wallpaper and never
//!   says why.
//! * **Accessibility** for input. Without it `CGEventPost` reports success and the
//!   events go nowhere, so the bot would decide at full speed and press nothing.
//!   `AXIsProcessTrusted` is checked for the same reason.
//!
//! Both are granted per application in System Settings, survive restarts, and are
//! cleared by `tccutil reset`. Some of what is needed works *before* the Screen
//! Recording grant: the window *list* is available, but `kCGWindowName` (the
//! title) is not, because macOS withholds other applications' titles. That is why
//! titles here fall back to the owning application's name - a list of untitled
//! windows is a list nobody can pick from.

use std::collections::HashSet;

use core_foundation::base::TCFType;
use core_foundation::string::{CFString, CFStringRef};
use core_graphics::color_space::CGColorSpace;
use core_graphics::context::CGContext;
use core_graphics::event::{CGEvent, CGEventTapLocation, CGEventType, CGMouseButton};
use core_graphics::event_source::{CGEventSource, CGEventSourceStateID};
use core_graphics::geometry::{CGPoint, CGRect, CGSize};
use core_graphics::window::{
    CGWindowID, copy_window_info, create_image, kCGNullWindowID, kCGWindowBounds,
    kCGWindowImageBoundsIgnoreFraming, kCGWindowImageNominalResolution, kCGWindowImageShouldBeOpaque,
    kCGWindowLayer, kCGWindowListOptionIncludingWindow, kCGWindowListOptionOnScreenOnly,
    kCGWindowName, kCGWindowNumber, kCGWindowOwnerName, kCGWindowOwnerPID,
};

use super::{Handle, WindowInfo};

/// Terminals, and the shells that run inside them. On macOS the shell is a child
/// of the terminal application, so the terminal's own name is what the process
/// tables reach.
const TERMINAL_PROCESSES: [&str; 11] = [
    "terminal",
    "iterm2",
    "iterm",
    "alacritty",
    "kitty",
    "wezterm",
    "wezterm-gui",
    "hyper",
    "tabby",
    "warp",
    "ghostty",
];

const APP_PROCESSES: [&str; 12] = [
    "safari",
    "google chrome",
    "firefox",
    "brave browser",
    "microsoft edge",
    "opera",
    "code",
    "code - insiders",
    "sublime_text",
    "discord",
    "slack",
    "zoom.us",
];

pub const SHELL_TITLES: [&str; 3] = ["program manager", "settings", "task manager"];

pub fn is_shell_process(process: &str) -> bool {
    TERMINAL_PROCESSES.contains(&process)
}

pub fn is_non_game_process(process: &str) -> bool {
    TERMINAL_PROCESSES.contains(&process) || APP_PROCESSES.contains(&process)
}

// ---- the two grants ----

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
}

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> bool;
}

/// Is the Screen Recording grant in place?
pub fn has_screen_recording() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

/// Is the Accessibility grant in place?
pub fn has_accessibility() -> bool {
    unsafe { AXIsProcessTrusted() }
}

pub fn is_supported() -> bool {
    // The APIs are always there; the grants decide whether what comes back is the
    // game or the wallpaper, and `capability_report` says which.
    true
}

pub fn capability_report() -> Vec<String> {
    let mut report = Vec::new();
    if has_screen_recording() {
        report.push("Screen Recording is granted: capture will see the window.".to_string());
    } else {
        report.push(
            "Screen Recording is NOT granted. The capture API will return the desktop \
             wallpaper and the menu bar instead of the window, with no error - the bot would \
             learn from a picture that contains none of the game. Grant it in System Settings \
             -> Privacy & Security -> Screen Recording, then restart the bot."
                .to_string(),
        );
        report.push(
            "The window list still works without it, though macOS withholds other \
             applications' window titles, so the titles shown come from the owning \
             application."
                .to_string(),
        );
    }
    if has_accessibility() {
        report.push("Accessibility is granted: input injection will reach the game.".to_string());
    } else {
        report.push(
            "Accessibility is NOT granted. CGEventPost will report success and the events will \
             go nowhere, so the bot would decide at full speed and press nothing. Grant it in \
             System Settings -> Privacy & Security -> Accessibility, then restart the bot."
                .to_string(),
        );
    }
    report
}

// ---- processes ----

#[link(name = "proc")]
unsafe extern "C" {
    fn proc_pidpath(pid: i32, buffer: *mut u8, buffersize: u32) -> i32;
}

/// Executable name of `pid`, from `libproc`.
fn process_path(pid: u32) -> String {
    let mut buffer = [0u8; 4096];
    let written = unsafe { proc_pidpath(pid as i32, buffer.as_mut_ptr(), buffer.len() as u32) };
    if written <= 0 {
        return String::new();
    }
    let path = String::from_utf8_lossy(&buffer[..written as usize]).to_string();
    path.rsplit('/').next().unwrap_or("").to_lowercase()
}

pub fn get_process_name(pid: u32) -> String {
    process_path(pid)
}

/// This process, and nothing else.
///
/// **A limitation worth stating.** Windows and Linux walk the parent chain to find
/// the terminal the bot was launched from, which is what keeps it from typing into
/// a shell. macOS has no equally cheap, layout-stable way to ask for a parent pid
/// (`sysctl`'s `kinfo_proc` layout varies between releases), so what protects a
/// Mac run is the *name* check: [`crate::platform::looks_like_game`] refuses any
/// window whose process is a known terminal, and every macOS terminal is on that
/// list. That is weaker than the parent chain, and strong enough for the hazard it
/// covers - the terminal this bot was started from.
pub fn own_process_ids() -> HashSet<u32> {
    let mut ids = HashSet::new();
    ids.insert(std::process::id());
    ids
}

pub fn window_is_own_process(handle: Handle) -> bool {
    let Some(pid) = window_pid(handle) else {
        return false;
    };
    own_process_ids().contains(&pid)
}

/// macOS has no console window: the terminal is an ordinary window, and the
/// process-name check above is what identifies it.
pub fn console_window() -> Handle {
    0
}

// ---- windows ----

/// One entry of the window list, before it is filtered.
struct RawWindow {
    id: u32,
    pid: u32,
    title: String,
    owner: String,
    layer: i64,
    bounds: (f64, f64, f64, f64),
}

/// `kCGMouseEventDeltaX` and `kCGMouseEventDeltaY`, from `CGEventTypes.h`: the
/// fields a mouse event carries its relative motion in.
const MOUSE_DELTA_X: u32 = 4;
const MOUSE_DELTA_Y: u32 = 5;

// The window list arrives as CoreFoundation collections, and it is walked with
// the C API rather than through the safe wrappers: the wrapper's key typing wants
// a conversion trait that does not line up with a dictionary keyed by raw
// pointers, and `CFDictionaryGetValue` is what the wrapper calls anyway.
//
// Every pointer here comes from the create-rule array `copy_window_info`
// returned, or from a `__get`-rule borrow of it, and none of them outlives it.
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFArrayGetCount(array: *const std::ffi::c_void) -> isize;
    fn CFArrayGetValueAtIndex(array: *const std::ffi::c_void, index: isize) -> *const std::ffi::c_void;
    fn CFDictionaryGetValue(
        dictionary: *const std::ffi::c_void,
        key: *const std::ffi::c_void,
    ) -> *const std::ffi::c_void;
    fn CFNumberGetValue(
        number: *const std::ffi::c_void,
        the_type: isize,
        value: *mut std::ffi::c_void,
    ) -> bool;
    fn CFStringGetCString(
        string: *const std::ffi::c_void,
        buffer: *mut u8,
        buffer_size: isize,
        encoding: u32,
    ) -> bool;
    fn CFRelease(value: *const std::ffi::c_void);
}

/// `kCFNumberSInt64Type` and `kCFStringEncodingUTF8`.
const CF_NUMBER_SINT64: isize = 4;
const CF_STRING_UTF8: u32 = 0x0800_0100;

/// One value out of a window-list dictionary.
fn dictionary_value(dictionary: *const std::ffi::c_void, key: CFStringRef) -> *const std::ffi::c_void {
    if dictionary.is_null() {
        return std::ptr::null();
    }
    unsafe { CFDictionaryGetValue(dictionary, key as *const std::ffi::c_void) }
}

fn number_value(dictionary: *const std::ffi::c_void, key: CFStringRef) -> Option<i64> {
    let value = dictionary_value(dictionary, key);
    if value.is_null() {
        return None;
    }
    let mut out: i64 = 0;
    let ok = unsafe {
        CFNumberGetValue(
            value,
            CF_NUMBER_SINT64,
            &mut out as *mut i64 as *mut std::ffi::c_void,
        )
    };
    ok.then_some(out)
}

fn string_value(dictionary: *const std::ffi::c_void, key: CFStringRef) -> Option<String> {
    let value = dictionary_value(dictionary, key);
    if value.is_null() {
        return None;
    }
    let mut buffer = [0u8; 1024];
    let ok = unsafe {
        CFStringGetCString(
            value,
            buffer.as_mut_ptr(),
            buffer.len() as isize,
            CF_STRING_UTF8,
        )
    };
    if !ok {
        return None;
    }
    let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(buffer.len());
    Some(String::from_utf8_lossy(&buffer[..end]).to_string())
}

/// The window's rectangle, out of the nested bounds dictionary.
fn bounds_value(dictionary: *const std::ffi::c_void) -> (f64, f64, f64, f64) {
    let key = unsafe { kCGWindowBounds };
    let bounds = dictionary_value(dictionary, key);
    if bounds.is_null() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let read = |name: &str| -> f64 {
        // The bounds dictionary is keyed by CFStrings like "X", "Y", "Width".
        let key = CFString::new(name);
        let value = dictionary_value(bounds, key.as_concrete_TypeRef());
        if value.is_null() {
            return 0.0;
        }
        let mut out: f64 = 0.0;
        let _ = unsafe {
            CFNumberGetValue(
                value,
                13, // kCFNumberFloat64Type
                &mut out as *mut f64 as *mut std::ffi::c_void,
            )
        };
        out
    };
    (read("X"), read("Y"), read("Width"), read("Height"))
}

/// Is this a window a user could be looking at, rather than a system overlay?
fn is_ordinary_window(raw: &RawWindow) -> bool {
    // Anything above layer 0 is the menu bar, the Dock, a floating panel or a
    // shadow: never what the user means by "the game".
    if raw.layer != 0 {
        return false;
    }
    let (_, _, width, height) = raw.bounds;
    width >= 1.0 && height >= 1.0
}

/// A window as [`crate::platform`] wants it.
fn into_window(raw: &RawWindow) -> WindowInfo {
    let (x, y, width, height) = raw.bounds;
    let title = if raw.title.trim().is_empty() {
        format!("{} (window {})", raw.owner, raw.id)
    } else {
        raw.title.clone()
    };
    WindowInfo {
        handle: raw.id as Handle,
        title,
        pid: raw.pid,
        process: get_process_name(raw.pid),
        rect: (
            x.round() as i32,
            y.round() as i32,
            width.round() as i32,
            height.round() as i32,
        ),
        client: (width.round() as i32, height.round() as i32),
        minimized: false,
        own: own_process_ids().contains(&raw.pid),
        console: false,
    }
}

pub fn enumerate_windows(min_area: i32) -> Vec<WindowInfo> {
    let mut found: Vec<WindowInfo> = raw_windows()
        .iter()
        .filter(|raw| is_ordinary_window(raw))
        .map(into_window)
        .filter(|window| window.rect.2.saturating_mul(window.rect.3) >= min_area)
        .collect();
    found.sort_by(|a, b| {
        let area = |w: &WindowInfo| w.rect.2.saturating_mul(w.rect.3);
        area(b).cmp(&area(a))
    });
    found
}

pub fn window_entry(handle: Handle) -> Option<WindowInfo> {
    raw_windows()
        .iter()
        .find(|raw| raw.id as Handle == handle)
        .map(into_window)
}

pub fn window_pid(handle: Handle) -> Option<u32> {
    window_entry(handle).map(|window| window.pid)
}

pub fn client_size(handle: Handle) -> (i32, i32) {
    window_entry(handle)
        .map(|window| window.client)
        .unwrap_or((0, 0))
}

pub fn is_minimized(_handle: Handle) -> bool {
    // A minimized window leaves the window list entirely, which is how a missing
    // entry already reports it.
    false
}

pub fn is_window(handle: Handle) -> bool {
    raw_windows().iter().any(|raw| raw.id as Handle == handle)
}

pub fn window_title(handle: Handle) -> String {
    window_entry(handle)
        .map(|window| window.title)
        .unwrap_or_default()
}

pub fn get_window_pid(handle: Handle) -> u32 {
    window_pid(handle).unwrap_or(0)
}

/// The window the user is looking at: the first ordinary window in the list,
/// which CoreGraphics returns front to back.
pub fn foreground_window() -> Handle {
    raw_windows()
        .iter()
        .find(|raw| is_ordinary_window(raw))
        .map(|raw| raw.id as Handle)
        .unwrap_or(0)
}
/// Every window the window server will describe, front to back.
///
/// The order matters: the list comes back front-most first, which is what makes
/// the frontmost window cheap to find.
fn raw_windows() -> Vec<RawWindow> {
    let Some(array) = copy_window_info(kCGWindowListOptionOnScreenOnly, kCGNullWindowID) else {
        return Vec::new();
    };
    let array_ref = array.as_concrete_TypeRef() as *const std::ffi::c_void;
    let count = unsafe { CFArrayGetCount(array_ref) };
    let mut out = Vec::new();
    for index in 0..count {
        let dictionary = unsafe { CFArrayGetValueAtIndex(array_ref, index) };
        if dictionary.is_null() {
            continue;
        }
        let (Some(id), Some(pid)) = (
            number_value(dictionary, unsafe { kCGWindowNumber }),
            number_value(dictionary, unsafe { kCGWindowOwnerPID }),
        ) else {
            continue;
        };
        out.push(RawWindow {
            id: id as u32,
            pid: pid as u32,
            title: string_value(dictionary, unsafe { kCGWindowName }).unwrap_or_default(),
            owner: string_value(dictionary, unsafe { kCGWindowOwnerName }).unwrap_or_default(),
            layer: number_value(dictionary, unsafe { kCGWindowLayer }).unwrap_or(0),
            bounds: bounds_value(dictionary),
        });
    }
    // The array came back under the create rule, so this owns a reference to it.
    unsafe { CFRelease(array_ref) };
    out
}
// =============================================================================
// Frame capture
// =============================================================================

/// `kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst`: four bytes per
/// pixel, blue first, which is the order the observation pipeline reads.
const BITMAP_BGRA: u32 = 2 | (2 << 12);

/// Captures a window with `CGWindowListCreateImage`.
pub struct FrameGrabber {
    handle: Handle,
    pixels: Vec<u8>,
    pub last_error: Option<String>,
    pub frames: u64,
    total_seconds: f64,
    warned: bool,
}

impl FrameGrabber {
    pub fn new(handle: Handle) -> Self {
        Self {
            handle,
            pixels: Vec::new(),
            last_error: None,
            frames: 0,
            total_seconds: 0.0,
            warned: false,
        }
    }

    pub fn grab(&mut self) -> Option<crate::capture::Frame> {
        let started = std::time::Instant::now();
        let result = self.grab_inner();
        self.frames += 1;
        self.total_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn grab_inner(&mut self) -> Option<crate::capture::Frame> {
        if !has_screen_recording() && !self.warned {
            self.warned = true;
            self.last_error = Some(
                "Screen Recording is not granted, so this capture returns the desktop rather \
                 than the window (see --platform)"
                    .to_string(),
            );
            println!("[Capture] {}", self.last_error.clone().unwrap_or_default());
        }
        let window = window_entry(self.handle)?;
        let (x, y, width, height) = (
            window.rect.0 as f64,
            window.rect.1 as f64,
            window.rect.2 as f64,
            window.rect.3 as f64,
        );
        // The window's own rectangle is passed as the bounds *and* the window is
        // named, so this is right under either reading of the API: with
        // `IncludingWindow` the image is the window, and if the rectangle is
        // honoured it is the window's own rectangle.
        let image = create_image(
            CGRect::new(&CGPoint::new(x, y), &CGSize::new(width, height)),
            kCGWindowListOptionIncludingWindow,
            self.handle as CGWindowID,
            kCGWindowImageBoundsIgnoreFraming
                | kCGWindowImageNominalResolution
                | kCGWindowImageShouldBeOpaque,
        )?;
        let (image_width, image_height) = (image.width(), image.height());
        if image_width == 0 || image_height == 0 {
            self.last_error = Some("the window produced an empty image".to_string());
            return None;
        }
        // Drawing into a context this code owns is what makes the pixel format
        // knowable: a `CGImage`'s own format depends on the window's backing
        // store, and guessing at it is how a capture comes back with swapped
        // channels.
        self.pixels.resize(image_width * image_height * 4, 0);
        let space = CGColorSpace::create_device_rgb();
        let context = CGContext::create_bitmap_context(
            Some(self.pixels.as_mut_ptr() as *mut std::ffi::c_void),
            image_width,
            image_height,
            8,
            image_width * 4,
            &space,
            BITMAP_BGRA,
        );
        context.draw_image(
            CGRect::new(
                &CGPoint::new(0.0, 0.0),
                &CGSize::new(image_width as f64, image_height as f64),
            ),
            &image,
        );
        // The context is dropped here, before the buffer is read: it must not
        // outlive the memory it was given.
        drop(context);
        for pixel in self.pixels.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
        self.last_error = None;
        Some(crate::capture::Frame::new(
            image_width,
            image_height,
            self.pixels.clone(),
        ))
    }

    pub fn mean_ms(&self) -> f64 {
        1000.0 * self.total_seconds / self.frames.max(1) as f64
    }
}

// =============================================================================
// Input injection
// =============================================================================

/// macOS virtual key codes for the names a keymap uses.
///
/// These are ANSI *positional* codes, not characters: the code for `W` is where
/// the W key sits on a US layout, which is what a game binds. A different
/// physical layout is a different keyboard, and remapping it is what the keyboard
/// preferences are for.
pub fn keycode_for(name: &str) -> Option<u16> {
    let upper = name.trim().to_uppercase();
    Some(match upper.as_str() {
        "A" => 0,
        "S" => 1,
        "D" => 2,
        "F" => 3,
        "H" => 4,
        "G" => 5,
        "Z" => 6,
        "X" => 7,
        "C" => 8,
        "V" => 9,
        "B" => 11,
        "Q" => 12,
        "W" => 13,
        "E" => 14,
        "R" => 15,
        "Y" => 16,
        "T" => 17,
        "1" => 18,
        "2" => 19,
        "3" => 20,
        "4" => 21,
        "6" => 22,
        "5" => 23,
        "=" => 24,
        "9" => 25,
        "7" => 26,
        "-" => 27,
        "8" => 28,
        "0" => 29,
        "]" => 30,
        "O" => 31,
        "U" => 32,
        "[" => 33,
        "I" => 34,
        "P" => 35,
        "ENTER" => 36,
        "L" => 37,
        "J" => 38,
        "'" => 39,
        "K" => 40,
        ";" => 41,
        "\\" => 42,
        "," => 43,
        "/" => 44,
        "N" => 45,
        "M" => 46,
        "." => 47,
        "TAB" => 48,
        "SPACE" => 49,
        "`" => 50,
        "BACKSPACE" => 51,
        "ESC" => 53,
        "F1" => 122,
        "F2" => 120,
        "F3" => 99,
        "F4" => 118,
        "F5" => 96,
        "F6" => 97,
        "F7" => 98,
        "F8" => 100,
        "F9" => 101,
        "F10" => 109,
        "F11" => 103,
        "F12" => 111,
        "F13" => 105,
        "F14" => 107,
        "F15" => 113,
        "HOME" => 115,
        "END" => 119,
        "PAGEUP" => 116,
        "PAGEDOWN" => 121,
        "DELETE" => 117,
        "UP" => 126,
        "DOWN" => 125,
        "LEFT" => 123,
        "RIGHT" => 124,
        "LSHIFT" | "SHIFT" => 56,
        "RSHIFT" => 60,
        "LCTRL" | "CTRL" => 59,
        "RCTRL" => 62,
        "LALT" | "ALT" => 58,
        "RALT" => 61,
        "WIN" => 55,
        _ => return None,
    })
}

/// The keymap's code for a name, as a macOS virtual key code.
///
/// The keymap file stores the Windows virtual-key code, because that is the
/// vocabulary the file is written in. The *name* is what travels; the number does
/// not.
pub fn keycode_from_code(vk: u32) -> Option<u16> {
    keycode_for(&crate::keys::key_name(vk, false))
}

/// Sends real keyboard and mouse input, through `CGEventPost`.
pub struct Injector {
    handle: Handle,
    source: Option<CGEventSource>,
    pub policy: crate::config::FocusPolicy,
    pub suspended: bool,
    pub enabled: bool,
    pub blocked: u64,
    pub failures: u64,
    pub skipped_unfocused: u64,
    pub events_sent: u64,
    pub keys_sent: u64,
    pub mouse_sent: u64,
    pub last_error: Option<String>,
    focused: bool,
    focus_attempted: bool,
    focus_warned: bool,
    focus_notice_at: std::time::Instant,
    pub transient_vks: HashSet<u32>,
    position: Option<(f64, f64)>,
    warned: bool,
}

impl Injector {
    pub fn new(handle: Handle, policy: crate::config::FocusPolicy) -> Self {
        Self {
            handle,
            source: CGEventSource::new(CGEventSourceStateID::CombinedSessionState).ok(),
            policy,
            suspended: false,
            enabled: true,
            blocked: 0,
            failures: 0,
            skipped_unfocused: 0,
            events_sent: 0,
            keys_sent: 0,
            mouse_sent: 0,
            last_error: None,
            focused: false,
            focus_attempted: false,
            focus_warned: false,
            focus_notice_at: std::time::Instant::now(),
            transient_vks: HashSet::new(),
            position: None,
            warned: false,
        }
    }

    pub fn focused(&self) -> bool {
        foreground_window() == self.handle
    }

    /// Bring the game window forward.
    ///
    /// macOS deliberately gives an application no way to raise another
    /// application's window without the Accessibility grant, and even with it the
    /// request is one the system may ignore. A refusal is reported, not fought:
    /// the honest instruction is "click the game".
    pub fn acquire_focus(&mut self, force: bool) -> bool {
        if self.focused() {
            self.focused = true;
            self.focus_warned = false;
            return true;
        }
        if self.policy == crate::config::FocusPolicy::Never {
            return false;
        }
        if !force && self.policy == crate::config::FocusPolicy::Once && self.focus_attempted {
            return false;
        }
        self.focus_attempted = true;
        // Raising another application's window means an Apple event to its bundle
        // id, which the window list does not carry. Left unimplemented rather than
        // faked: a wrong request is worse than none.
        self.last_error = Some(if has_accessibility() {
            "bringing the window forward is not implemented on macOS yet; click the game"
                .to_string()
        } else {
            "macOS will not let this process bring another application forward without the \
             Accessibility grant; click the game yourself"
                .to_string()
        });
        false
    }

    pub fn begin_action(&mut self) {
        self.focused = self.focused();
        if self.focused {
            if self.focus_warned {
                self.focus_warned = false;
                println!("[Input] The game window is focused again; input resumed.");
            }
            return;
        }
        if self.policy == crate::config::FocusPolicy::Always && self.acquire_focus(true) {
            return;
        }
        if !self.focus_warned {
            self.focus_warned = true;
            self.focus_notice_at = std::time::Instant::now();
            println!(
                "[Input] The game window is not focused, so nothing is being sent. Click the \
                 game - input resumes on its own. (The terminal can be used normally \
                 meanwhile.)"
            );
        } else if self.focus_notice_at.elapsed().as_secs_f64() >= 60.0 {
            self.focus_notice_at = std::time::Instant::now();
            println!(
                "[Input] Still sending nothing: the game window has not been the front window \
                 for {} step(s). Click the game to let the bot play it.",
                self.skipped_unfocused
            );
        }
    }

    pub fn end_action(&mut self) {
        self.focused = false;
    }

    pub fn status_line(&self) -> String {
        let mut line = format!(
            "[Input] game window {} - {} key event(s) and {} mouse event(s) delivered, \
             {} step(s) skipped while unfocused",
            if self.focused() { "focused" } else { "NOT focused" },
            self.keys_sent,
            self.mouse_sent,
            self.skipped_unfocused
        );
        if self.failures > 0 {
            line += &format!(", {} inject failure(s)", self.failures);
        }
        if let Some(error) = &self.last_error {
            line += &format!(" ({error})");
        }
        line
    }

    fn may_inject(&mut self, what: &str, forcing: bool) -> bool {
        if !has_accessibility() {
            if !self.warned {
                self.warned = true;
                self.failures += 1;
                println!(
                    "[Input] BLOCKED injection ({what}): Accessibility is not granted, so \
                     CGEventPost would report success and deliver nothing. See --platform."
                );
            }
            return false;
        }
        if !self.enabled {
            self.blocked += 1;
            if self.blocked <= 5 {
                println!(
                    "[Input] BLOCKED injection ({what}): input is disabled in this mode. This \
                     is a bug."
                );
            }
            return false;
        }
        if self.suspended {
            return false;
        }
        if !forcing && !self.focused {
            self.skipped_unfocused += 1;
            return false;
        }
        true
    }

    fn post(&mut self, event: CGEvent) {
        event.post(CGEventTapLocation::HID);
        self.events_sent += 1;
    }

    fn send_key(&mut self, code: u16, down: bool, forcing: bool) {
        let what = format!("key {code} {}", if down { "down" } else { "up" });
        if !self.may_inject(&what, forcing) {
            return;
        }
        let Some(source) = self.source.clone() else {
            self.failures += 1;
            self.last_error = Some("no event source".to_string());
            return;
        };
        match CGEvent::new_keyboard_event(source, code, down) {
            Ok(event) => {
                self.post(event);
                self.keys_sent += 1;
            }
            Err(_) => {
                self.failures += 1;
                self.last_error = Some("CGEventCreateKeyboardEvent failed".to_string());
            }
        }
    }

    fn send_button(&mut self, button: CGMouseButton, down: bool, forcing: bool) {
        let what = format!("button {button:?} {}", if down { "down" } else { "up" });
        if !self.may_inject(&what, forcing) {
            return;
        }
        let kind = match (button, down) {
            (CGMouseButton::Left, true) => CGEventType::LeftMouseDown,
            (CGMouseButton::Left, false) => CGEventType::LeftMouseUp,
            (CGMouseButton::Right, true) => CGEventType::RightMouseDown,
            (CGMouseButton::Right, false) => CGEventType::RightMouseUp,
            (_, true) => CGEventType::OtherMouseDown,
            (_, false) => CGEventType::OtherMouseUp,
        };
        let Some(source) = self.source.clone() else {
            self.failures += 1;
            return;
        };
        let (x, y) = self.position.unwrap_or((0.0, 0.0));
        match CGEvent::new_mouse_event(source, kind, CGPoint::new(x, y), button) {
            Ok(event) => {
                self.post(event);
                self.mouse_sent += 1;
            }
            Err(_) => {
                self.failures += 1;
                self.last_error = Some("CGEventCreateMouseEvent failed".to_string());
            }
        }
    }

    /// Relative mouse motion.
    ///
    /// Unlike X11 this can be genuinely relative: a `MouseMoved` event carries
    /// `kCGMouseEventDeltaX/Y` fields, and the point it names is only where the
    /// pointer should end up. Both are set, because games differ in which of the
    /// two they read.
    pub fn mouse_move(&mut self, dx: i32, dy: i32) {
        if !self.may_inject("mouse move", false) {
            return;
        }
        let Some(source) = self.source.clone() else {
            return;
        };
        let (x, y) = self.position.unwrap_or((0.0, 0.0));
        let (next_x, next_y) = (x + dx as f64, y + dy as f64);
        self.position = Some((next_x, next_y));
        match CGEvent::new_mouse_event(
            source,
            CGEventType::MouseMoved,
            CGPoint::new(next_x, next_y),
            CGMouseButton::Left,
        ) {
            Ok(event) => {
                event.set_integer_value_field(MOUSE_DELTA_X, dx as i64);
                event.set_integer_value_field(MOUSE_DELTA_Y, dy as i64);
                self.post(event);
                self.mouse_sent += 1;
            }
            Err(_) => {
                self.failures += 1;
                self.last_error = Some("CGEventCreateMouseEvent failed".to_string());
            }
        }
    }

    pub fn press_vk(&mut self, vk: u32) {
        if let Some(code) = keycode_from_code(vk) {
            self.send_key(code, true, false);
        }
    }

    pub fn release_vk(&mut self, vk: u32) {
        // Forced, as on the other platforms: never leave a key down.
        if let Some(code) = keycode_from_code(vk) {
            self.send_key(code, false, true);
        }
    }

    pub fn tap_vk(&mut self, vk: u32, seconds: f32) {
        let Some(code) = keycode_from_code(vk) else {
            return;
        };
        self.transient_vks.insert(vk);
        self.send_key(code, true, false);
        std::thread::sleep(std::time::Duration::from_secs_f32(seconds.max(0.0)));
        self.send_key(code, false, false);
        self.transient_vks.remove(&vk);
    }

    pub fn mouse_down(&mut self, button: &str) {
        if let Some(code) = mouse_button(button) {
            self.send_button(code, true, false);
        }
    }

    pub fn mouse_up(&mut self, button: &str) {
        if let Some(code) = mouse_button(button) {
            self.send_button(code, false, true);
        }
    }

    pub fn click(&mut self, button: &str, seconds: f32) {
        self.mouse_down(button);
        std::thread::sleep(std::time::Duration::from_secs_f32(seconds.max(0.0)));
        self.mouse_up(button);
    }

    pub fn release_all(&mut self, held_vks: &[u32]) {
        let held: Vec<u32> = held_vks.to_vec();
        for vk in held {
            self.release_vk(vk);
        }
    }
}

fn mouse_button(button: &str) -> Option<CGMouseButton> {
    match button {
        "left" => Some(CGMouseButton::Left),
        "right" => Some(CGMouseButton::Right),
        "middle" => Some(CGMouseButton::Center),
        _ => None,
    }
}

// =============================================================================
// Global hotkeys, and the calibration recorder
// =============================================================================

/// Global hotkeys, not built yet.
///
/// `CGEventTap` is the API - a listen-only tap for the recorder and a default tap
/// for the hotkeys - and neither is exposed by the bindings used here, so this
/// means hand-written FFI for the tap callback (and a run loop to deliver it).
/// Both also need the Accessibility grant. Until then a Mac run starts
/// immediately and is stopped with Ctrl+C, which the controller already handles.
pub struct Hotkeys {
    messages: Vec<String>,
}

impl Hotkeys {
    pub fn new(_pause: &str, _save: &str, _quit: &str) -> Self {
        Self {
            messages: vec![
                "Global hotkeys need the Accessibility grant and a CGEventTap, which this build \
                 does not install yet; Ctrl+C saves and quits."
                    .to_string(),
            ],
        }
    }

    pub fn start(&mut self) -> bool {
        false
    }

    pub fn available(&self) -> bool {
        false
    }

    pub fn poll(&mut self) -> Vec<super::HotkeyAction> {
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

/// The calibration recorder, not built yet for the same reason.
pub struct KeyRecorder {
    announced: bool,
}

impl KeyRecorder {
    pub fn start(_toggle_vk: u32, _target: Handle, _only_when_focused: bool) -> Self {
        Self { announced: false }
    }

    pub fn installed(&mut self) -> bool {
        if !self.announced {
            self.announced = true;
            println!(
                "[Calibrate] macOS key recording is not built yet: it needs a listen-only \
                 CGEventTap, which also requires the Accessibility grant. Write or edit \
                 keymap.json by hand and the bot will use it."
            );
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

    pub fn snapshot(&self, _target: Option<&WindowInfo>) -> crate::keys::Recording {
        crate::keys::Recording::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keycodes_follow_the_ansi_layout() {
        // W is 13, A is 0, S is 1, D is 2: the positional codes a game binds.
        assert_eq!(keycode_for("W"), Some(13));
        assert_eq!(keycode_for("A"), Some(0));
        assert_eq!(keycode_for("S"), Some(1));
        assert_eq!(keycode_for("D"), Some(2));
        assert_eq!(keycode_for("SPACE"), Some(49));
        assert_eq!(keycode_for("F8"), Some(100));
        assert_eq!(keycode_for("LEFT"), Some(123));
        // A mouse button is not a keycode, and is refused rather than guessed.
        assert_eq!(keycode_for("MOUSE_LEFT"), None);
    }

    #[test]
    fn the_keymap_vocabulary_reaches_a_keycode() {
        // The keymap stores Windows virtual-key codes; the *name* travels.
        assert_eq!(keycode_from_code(0x57), Some(13), "VK_W is the W key");
        assert_eq!(keycode_from_code(0x20), Some(49), "VK_SPACE is space");
        assert_eq!(keycode_from_code(0x77), Some(100), "VK_F8 is F8");
        assert_eq!(keycode_from_code(0x01), None, "a mouse VK is not a keycode");
    }

    #[test]
    fn this_process_is_its_own_only_member() {
        let own = own_process_ids();
        assert_eq!(own.len(), 1);
        assert!(own.contains(&std::process::id()));
    }

    #[test]
    fn the_macos_process_tables_name_the_terminals() {
        assert!(is_shell_process("iterm2"));
        assert!(is_non_game_process("google chrome"));
        assert!(!is_non_game_process("somegame"));
    }
}
