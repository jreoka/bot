//! The Win32 mechanism behind [`crate::platform`].
//!
//! Input goes to a *window*, so the first thing a run does is decide which one.
//! It will not simply take the largest window on the desktop, because that is
//! usually the terminal it was launched from - and driving your own terminal
//! means typing the bot's actions into a shell prompt while the game sits
//! untouched, at twenty decisions a second, with no error printed anywhere.
//!
//! Two things are marked here, once, and every consumer agrees about them: which
//! windows belong to this process or to the shell that launched it (never
//! driven), and which executables are known non-games (never auto-selected).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::{CloseHandle, HWND, LPARAM, MAX_PATH, POINT, RECT, WPARAM};
use windows::Win32::System::Console::GetConsoleWindow;
use windows::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
};
use windows::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
    QueryFullProcessImageNameW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, ClipCursor, EnumWindows, GetClientRect, GetCursorPos, GetForegroundWindow,
    GetWindowRect, GetWindowTextW, GetWindowThreadProcessId, HHOOK, IsIconic, IsWindow,
    IsWindowVisible, SetCursorPos,
};
use windows::core::PWSTR;

use super::{Handle, HotkeyAction, WindowInfo};

/// Windows that are almost always the desktop shell rather than a game. Kept
/// short deliberately: a false negative (refusing a real game) is worse than a
/// false positive (offering a picker).
const SHELL_PROCESSES: [&str; 9] = [
    "explorer.exe",
    "searchhost.exe",
    "startmenuexperiencehost.exe",
    "shellexperiencehost.exe",
    "textinputhost.exe",
    "dwm.exe",
    "applicationframehost.exe",
    "taskmgr.exe",
    "systemsettings.exe",
];

pub const SHELL_TITLES: [&str; 3] = ["program manager", "settings", "task manager"];

/// The group that matters: the terminal, IDE or shell the bot was *launched
/// from*. A bot that drives its own terminal types its actions into a shell
/// prompt, and because the injector only checks that the target is the
/// *foreground* window, that run looks perfectly healthy from the inside - 20
/// decisions a second, no error, no warning - while the game sits untouched.
const TERMINAL_PROCESSES: [&str; 17] = [
    "windowsterminal.exe",
    "openconsole.exe",
    "conhost.exe",
    "wt.exe",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "bash.exe",
    "wsl.exe",
    "mintty.exe",
    "putty.exe",
    "alacritty.exe",
    "wezterm-gui.exe",
    "conemu.exe",
    "conemu64.exe",
    "tabby.exe",
    "hyper.exe",
];

/// A guess rather than a hazard: driving a browser or an editor is merely the
/// wrong window, not a way to run a command. Excluded from auto-detection (a
/// maximised browser is otherwise the largest window on most desktops) and
/// warned about, but never refused.
const APP_PROCESSES: [&str; 21] = [
    "chrome.exe",
    "msedge.exe",
    "firefox.exe",
    "brave.exe",
    "opera.exe",
    "vivaldi.exe",
    "iexplore.exe",
    "code.exe",
    "code - insiders.exe",
    "devenv.exe",
    "notepad++.exe",
    "sublime_text.exe",
    "pycharm64.exe",
    "idea64.exe",
    "rider64.exe",
    "discord.exe",
    "slack.exe",
    "teams.exe",
    "ms-teams.exe",
    "zoom.exe",
    "spotify.exe",
];

pub fn is_shell_process(process: &str) -> bool {
    SHELL_PROCESSES.contains(&process)
}

pub fn is_non_game_process(process: &str) -> bool {
    TERMINAL_PROCESSES.contains(&process) || APP_PROCESSES.contains(&process)
}

pub fn is_supported() -> bool {
    true
}

pub fn capability_report() -> Vec<String> {
    vec![
        "Cursor confinement is available: while the bot moves the mouse or clicks, the \
         cursor is kept inside the game window, so a game that frees it (an inventory) \
         cannot have the click land outside. --no-clip-cursor turns it off."
            .to_string(),
    ]
}

fn window_text(hwnd: HWND) -> String {
    let mut buffer = [0u16; 512];
    let length = unsafe { GetWindowTextW(hwnd, &mut buffer) };
    if length <= 0 {
        return String::new();
    }
    String::from_utf16_lossy(&buffer[..length as usize])
}

pub fn window_title(handle: Handle) -> String {
    if handle == 0 {
        return String::new();
    }
    window_text(HWND(handle as *mut _))
}

pub fn get_window_pid(handle: Handle) -> u32 {
    let mut pid = 0u32;
    unsafe {
        GetWindowThreadProcessId(HWND(handle as *mut _), Some(&mut pid));
    }
    pid
}

// =============================================================================
// Frame capture
// =============================================================================

/// Reusable `PrintWindow` grabber.
///
/// The naive version allocates a device context and a bitmap per call, which
/// caps the loop at a few frames per second before any model runs at all. This
/// keeps one DC and one bitmap alive for the window's current client size and
/// only rebuilds them on resize, so the per-frame cost is `PrintWindow` itself.
pub struct FrameGrabber {
    handle: Handle,
    size: (i32, i32),
    window_dc: windows::Win32::Graphics::Gdi::HDC,
    memory_dc: windows::Win32::Graphics::Gdi::HDC,
    bitmap: windows::Win32::Graphics::Gdi::HBITMAP,
    /// What the memory DC had selected before the bitmap went in, so the bitmap
    /// can be taken back out for `GetDIBits` and put back for `PrintWindow`.
    default_bitmap: windows::Win32::Graphics::Gdi::HGDIOBJ,
    pixels: Vec<u8>,
    pub last_error: Option<String>,
    pub frames: u64,
    total_seconds: f64,
}

/// `PW_RENDERFULLCONTENT`: ask the window to render itself even when it is
/// using hardware acceleration, which plain `PrintWindow` cannot do.
const PW_RENDERFULLCONTENT: u32 = 0x0000_0002;

impl FrameGrabber {
    pub fn new(handle: Handle) -> Self {
        Self {
            handle,
            size: (0, 0),
            window_dc: Default::default(),
            memory_dc: Default::default(),
            bitmap: Default::default(),
            default_bitmap: Default::default(),
            pixels: Vec::new(),
            last_error: None,
            frames: 0,
            total_seconds: 0.0,
        }
    }

    fn release(&mut self) {
        use windows::Win32::Graphics::Gdi::{DeleteDC, DeleteObject, ReleaseDC};
        unsafe {
            if !self.bitmap.is_invalid() {
                let _ = DeleteObject(self.bitmap.into());
                self.bitmap = Default::default();
            }
            if !self.memory_dc.is_invalid() {
                let _ = DeleteDC(self.memory_dc);
                self.memory_dc = Default::default();
            }
            if !self.window_dc.is_invalid() {
                let _ = ReleaseDC(Some(HWND(self.handle as *mut _)), self.window_dc);
                self.window_dc = Default::default();
            }
        }
        self.size = (0, 0);
    }

    fn ensure_buffers(&mut self, width: i32, height: i32) {
        use windows::Win32::Graphics::Gdi::{
            CreateCompatibleBitmap, CreateCompatibleDC, GetWindowDC, SelectObject,
        };
        if (width, height) == self.size {
            return;
        }
        self.release();
        unsafe {
            self.window_dc = GetWindowDC(Some(HWND(self.handle as *mut _)));
            self.memory_dc = CreateCompatibleDC(Some(self.window_dc));
            // From the *window* DC, so the bitmap is colour: creating it from the
            // memory DC gives a one-bit monochrome bitmap, which captures
            // nothing but black and white.
            self.bitmap = CreateCompatibleBitmap(self.window_dc, width, height);
            // It has to be selected while `PrintWindow` draws, or the window
            // renders into the DC's default one-pixel bitmap and every frame
            // comes back black.
            self.default_bitmap = SelectObject(self.memory_dc, self.bitmap.into());
        }
        self.pixels = vec![0u8; (width as usize) * (height as usize) * 4];
        self.size = (width, height);
    }

    /// Capture the client area as BGRA, or `None`.
    ///
    /// A window that is closed, minimized or refusing `PrintWindow` returns
    /// `None` rather than raising: the caller decides what that means.
    pub fn grab(&mut self) -> Option<crate::capture::Frame> {
        use windows::Win32::Graphics::Gdi::{
            BI_RGB, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, GetDIBits, SelectObject,
        };
        use windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};

        let started = std::time::Instant::now();
        let result = (|| -> Option<crate::capture::Frame> {
            let (width, height) = client_size(self.handle);
            if width <= 0 || height <= 0 {
                self.last_error = Some("window has no client area".to_string());
                return None;
            }
            self.ensure_buffers(width, height);

            let mut printed = unsafe {
                PrintWindow(
                    HWND(self.handle as *mut _),
                    self.memory_dc,
                    PRINT_WINDOW_FLAGS(PW_RENDERFULLCONTENT),
                )
            };
            if !printed.as_bool() {
                // Some games only render into a plain PrintWindow.
                printed = unsafe {
                    PrintWindow(
                        HWND(self.handle as *mut _),
                        self.memory_dc,
                        PRINT_WINDOW_FLAGS(0),
                    )
                };
            }
            if !printed.as_bool() {
                self.last_error = Some(format!("PrintWindow returned {}", printed.0));
                return None;
            }

            // A top-down DIB (negative height) so the rows arrive in the order a
            // screen is read. `GetBitmapBits` would have been shorter, but it
            // hands back the device's own row order and says nothing about it.
            let mut info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width,
                    biHeight: -height,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            // `GetDIBits` requires the bitmap not to be selected into a device
            // context, so it comes back out for the copy and goes back in for
            // the next frame.
            let bitmap = unsafe { SelectObject(self.memory_dc, self.default_bitmap) };
            let copied = unsafe {
                GetDIBits(
                    self.memory_dc,
                    self.bitmap,
                    0,
                    height as u32,
                    Some(self.pixels.as_mut_ptr() as *mut _),
                    &mut info,
                    DIB_RGB_COLORS,
                )
            };
            let _ = unsafe { SelectObject(self.memory_dc, bitmap) };
            if copied == 0 {
                self.last_error = Some("GetDIBits copied no scan lines".to_string());
                return None;
            }
            self.last_error = None;
            Some(crate::capture::Frame::new(
                width as usize,
                height as usize,
                self.pixels.clone(),
            ))
        })();
        self.frames += 1;
        self.total_seconds += started.elapsed().as_secs_f64();
        result
    }

    pub fn mean_ms(&self) -> f64 {
        1000.0 * self.total_seconds / self.frames.max(1) as f64
    }
}

impl Drop for FrameGrabber {
    fn drop(&mut self) {
        self.release();
    }
}

// =============================================================================
// Input injection
// =============================================================================

/// Which keys need the extended-key flag, or they register as the
/// numeric-keypad equivalent.
const EXTENDED_VKS: [u32; 15] = [
    0xA1, 0xA3, 0xA5, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22, 0x25, 0x26, 0x27, 0x28, 0x5B, 0x5C,
];

/// Sends real keyboard and mouse input to the focused window.
///
/// Mouse-look is deliberately a real cursor delta through `SendInput`: games
/// read look from raw relative mouse motion, so a message-based approach cannot
/// turn the view at all. That also means the game window must be focused, which
/// is why this object watches the foreground window rather than assuming it.
///
/// **Focus is checked, not fought for.** Input is only sent while the game is the
/// foreground window; the policy decides whether the bot may take focus at all
/// ("once" - when a run starts or resumes; "always" - on every action, which is
/// what makes the terminal and the game wrestle for the keyboard; "never").
/// Sending keys at a window that is not in front does not reach the game - it
/// goes to whatever *is* in front, which for a bot started from a terminal means
/// the terminal itself.
pub struct Injector {
    handle: Handle,
    pub policy: crate::config::FocusPolicy,
    /// The soft switch used while the user has the controls: while it is set,
    /// this object injects nothing and says nothing, because it is expected
    /// rather than a bug.
    pub suspended: bool,
    /// The hard switch used by calibration, where any injection at all would be
    /// a bug worth shouting about.
    pub enabled: bool,
    pub blocked: u64,
    pub failures: u64,
    pub skipped_unfocused: u64,
    /// What actually left this process. Counting the events Windows accepted is
    /// the only way to tell "the bot is deciding" apart from "the bot is
    /// playing"; a run that decides 20 times a second while delivering nothing
    /// looks identical in every other number.
    pub events_sent: u64,
    pub keys_sent: u64,
    pub mouse_sent: u64,
    pub last_error: Option<String>,
    focused: bool,
    focus_attempted: bool,
    focus_warned: bool,
    focus_notice_at: std::time::Instant,
    pub transient_vks: HashSet<u32>,
    /// Whether the cursor is confined to the game window while the bot moves it
    /// or clicks. See [`crate::Config::clip_cursor`].
    pub clip_cursor: bool,
    /// Whether *this* injector is holding the cursor right now. The release is
    /// conditional on purpose: `ClipCursor(NULL)` is desktop-wide, so releasing
    /// a clip this injector never took would throw away one the game is holding.
    confined: bool,
    clip_failures: u64,
    clip_notice: bool,
}

impl Injector {
    pub fn new(handle: Handle, policy: crate::config::FocusPolicy, clip_cursor: bool) -> Self {
        Self {
            handle,
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
            clip_cursor,
            confined: false,
            clip_failures: 0,
            clip_notice: false,
        }
    }

    /// Would Windows deliver injected input to the game window?
    pub fn focused(&self) -> bool {
        foreground_window() == self.handle
    }

    /// Bring the game window forward. Called at start and on resume, not per
    /// step, so the desktop is not repeatedly yanked out from under the user.
    ///
    /// `force` asks for one attempt even under the "once" policy.
    pub fn acquire_focus(&mut self, force: bool) -> bool {
        use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
        use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;

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
        let target = HWND(self.handle as *mut _);
        if unsafe { SetForegroundWindow(target) }.as_bool() {
            self.focused = self.focused();
            if self.focused {
                self.focus_warned = false;
            }
            return self.focused;
        }
        // Windows refuses SetForegroundWindow from a process that does not own
        // the foreground; attaching to the foreground thread's input queue is
        // the standard way around it.
        let foreground = HWND(foreground_window() as *mut _);
        let mut pid = 0u32;
        let foreground_thread = unsafe { GetWindowThreadProcessId(foreground, Some(&mut pid)) };
        let own_thread = unsafe { GetCurrentThreadId() };
        let attached = unsafe { AttachThreadInput(own_thread, foreground_thread, true).as_bool() };
        let result = unsafe { SetForegroundWindow(target) }.as_bool();
        // Always detach: a leaked attachment couples this thread's input queue
        // to the other window's, and focus then behaves strangely for every
        // window afterwards.
        if attached {
            unsafe {
                let _ = AttachThreadInput(own_thread, foreground_thread, false);
            }
        }
        if !result {
            return false;
        }
        self.focused = self.focused();
        if self.focused {
            self.focus_warned = false;
        }
        self.focused
    }

    /// Called before a batch of input; decides whether it may be sent.
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
            // Repeating it once a minute matters: "the bot is running and
            // nothing is happening" is this, and the first message is easy to
            // miss in a wall of startup output.
            self.focus_notice_at = std::time::Instant::now();
            println!(
                "[Input] Still sending nothing: the game window has not been the front window \
                 for {} step(s). Click the game to let the bot play it.",
                self.skipped_unfocused
            );
        }
    }

    pub fn end_action(&mut self) {
        // The other half of the confinement claimed by a mouse move: whatever
        // this batch took, it gives back as soon as it is done injecting. The
        // cursor is the user's, and a bot that holds it through the gap between
        // decisions is a bot that has taken their mouse away.
        self.release_cursor_clip();
        self.focused = false;
    }

    /// Whether this injector may take the cursor right now.
    ///
    /// The same conditions `may_inject` applies, read without side effects: no
    /// confinement when the feature is off, when injection is disabled, when the
    /// user has the controls, or when the game is not the window the input would
    /// reach - the user's cursor must not be trapped by a batch that is going to
    /// be refused anyway.
    fn may_confine(&self) -> bool {
        self.clip_cursor && self.enabled && !self.suspended && self.focused()
    }

    /// Confine the cursor to the game window for the rest of this action.
    ///
    /// A game frees the cursor when it opens a menu - an inventory, a pause
    /// screen - and then reads the *real* pointer rather than a raw delta. The
    /// bot's own mouse steps are relative and unclamped, so the pointer walks
    /// out of the window within a second and the next click lands on whatever is
    /// under it. `ClipCursor` is the OS-level answer, and the same one GLFW uses
    /// for its disabled-cursor mode.
    ///
    /// Re-asserted before every move rather than tracked: the game resets the
    /// clip whenever it takes or gives up the cursor, so "already confined" is
    /// not something this side can know.
    fn confine_cursor(&mut self) {
        let Some(rect) = client_rect_on_screen(self.handle) else {
            self.note_clip_failure("the window has no client area to confine it to");
            return;
        };
        if let Err(error) = unsafe { ClipCursor(Some(&rect)) } {
            self.note_clip_failure(&format!("ClipCursor was refused ({error})"));
            return;
        }
        self.confined = true;
        // A clip only governs where the cursor may *go* next, and a pointer that
        // is already outside the rectangle - the user moved it while the bot was
        // paused, or it was never over the game - would sit there until
        // something moved it. The click about to be injected would land outside
        // the game, which is the exact failure this exists to stop, so the
        // pointer is pulled to the middle of the window explicitly.
        let mut pointer = POINT::default();
        if unsafe { GetCursorPos(&mut pointer) }.is_ok()
            && (pointer.x < rect.left
                || pointer.x >= rect.right
                || pointer.y < rect.top
                || pointer.y >= rect.bottom)
        {
            let centre_x = (rect.left + rect.right) / 2;
            let centre_y = (rect.top + rect.bottom) / 2;
            let _ = unsafe { SetCursorPos(centre_x, centre_y) };
        }
        if !self.clip_notice {
            self.clip_notice = true;
            println!(
                "[Input] Cursor confined to the game window while the bot moves it \
                 (--no-clip-cursor turns this off)."
            );
        }
    }

    /// A confinement failure is a warning, not a stop: the move still reaches the
    /// game, it is only the pointer that can still leave the window. Said once,
    /// because it would otherwise repeat twenty times a second.
    fn note_clip_failure(&mut self, reason: &str) {
        self.clip_failures += 1;
        if self.clip_failures == 1 {
            println!(
                "[Input] The cursor is NOT confined to the game window ({reason}), so a click \
                 can still land outside it."
            );
        }
    }

    /// Give the cursor back, if this injector is holding it.
    pub fn release_cursor_clip(&mut self) {
        if !self.confined {
            return;
        }
        self.confined = false;
        if let Err(error) = unsafe { ClipCursor(None) } {
            self.last_error = Some(format!("ClipCursor(NULL) failed ({error})"));
        }
    }

    /// One line: is input reaching a window, and how much of it.
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
            line += &format!(", {} SendInput failure(s)", self.failures);
        }
        if self.clip_failures > 0 {
            // The count is the only running evidence for a one-time warning: a
            // cursor that cannot be confined is a click that can land outside.
            line += &format!(", {} cursor-clip failure(s)", self.clip_failures);
        }
        if let Some(error) = &self.last_error {
            line += &format!(" ({error})");
        }
        line
    }

    fn may_inject(&mut self, what: &str, forcing: bool) -> bool {
        if !self.enabled {
            self.blocked += 1;
            if self.blocked <= 5 {
                println!(
                    "[Input] BLOCKED injection ({what}): input is disabled in this mode. \
                     This is a bug."
                );
            }
            return false;
        }
        if self.suspended {
            return false;
        }
        if !forcing && !self.focused {
            // No focus, no input: a key sent now would land in some other
            // window, and the game would never see it.
            self.skipped_unfocused += 1;
            return false;
        }
        true
    }

    fn send_key(&mut self, vk: u32, up: bool, forcing: bool) {
        use windows::Win32::UI::Input::KeyboardAndMouse::{
            INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYBD_EVENT_FLAGS, KEYEVENTF_EXTENDEDKEY,
            KEYEVENTF_KEYUP, MAPVK_VK_TO_VSC, MapVirtualKeyW, SendInput, VIRTUAL_KEY,
        };
        if !self.may_inject(&format!("vk 0x{vk:02X} {}", if up { "up" } else { "down" }), forcing) {
            return;
        }
        let scan = unsafe { MapVirtualKeyW(vk, MAPVK_VK_TO_VSC) } as u16;
        let mut flags = KEYBD_EVENT_FLAGS(0);
        if up {
            flags |= KEYEVENTF_KEYUP;
        }
        if EXTENDED_VKS.contains(&vk) {
            flags |= KEYEVENTF_EXTENDEDKEY;
        }
        let input = INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(vk as u16),
                    wScan: scan,
                    dwFlags: flags,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        let sent = unsafe {
            SendInput(&[input], std::mem::size_of::<INPUT>() as i32)
        };
        if sent == 0 {
            self.failures += 1;
            self.last_error = Some("SendInput(key) failed".to_string());
        } else {
            self.keys_sent += 1;
            self.events_sent += 1;
        }
    }

    fn send_mouse(&mut self, dx: i32, dy: i32, flags: u32, forcing: bool) {
        use windows::Win32::UI::Input::KeyboardAndMouse::{
            INPUT, INPUT_0, INPUT_MOUSE, MOUSEINPUT, MOUSE_EVENT_FLAGS, SendInput,
        };
        if !self.may_inject(&format!("mouse dx={dx} dy={dy} flags=0x{flags:04X}"), forcing) {
            return;
        }
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx,
                    dy,
                    mouseData: 0,
                    dwFlags: MOUSE_EVENT_FLAGS(flags),
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        let sent = unsafe {
            SendInput(&[input], std::mem::size_of::<INPUT>() as i32)
        };
        if sent == 0 {
            self.failures += 1;
            self.last_error = Some("SendInput(mouse) failed".to_string());
        } else {
            self.mouse_sent += 1;
            self.events_sent += 1;
        }
    }

    pub fn press_vk(&mut self, vk: u32) {
        if vk != 0 {
            self.send_key(vk, false, false);
        }
    }

    pub fn release_vk(&mut self, vk: u32) {
        // Releases are always sent, focused or not: a key left down is worse
        // than a key-up delivered to the wrong window (a key-up for a key that
        // is not down does nothing).
        if vk != 0 {
            self.send_key(vk, true, true);
        }
    }

    /// Down, hold briefly, up. The key is noted while it is physically down.
    pub fn tap_vk(&mut self, vk: u32, seconds: f32) {
        if vk == 0 {
            return;
        }
        // A mouse button taps as a click *wherever the cursor happens to be*, so
        // it confines the cursor first, exactly as a move does. The keyboard
        // path is where the run sends its clicks (`click:left` and friends are
        // virtual keys in the action space), which is why this check is here and
        // not only in `mouse_down`.
        if crate::keys::is_mouse_vk(vk) && self.may_confine() {
            self.confine_cursor();
        }
        self.transient_vks.insert(vk);
        self.send_key(vk, false, false);
        std::thread::sleep(std::time::Duration::from_secs_f32(seconds.max(0.0)));
        self.send_key(vk, true, false);
        self.transient_vks.remove(&vk);
    }

    pub fn mouse_move(&mut self, dx: i32, dy: i32) {
        // Confined *before* the delta is queued: the clamp is applied when the
        // system processes the move, so a clip taken after `SendInput` returns
        // can be too late to catch it.
        if self.may_confine() {
            self.confine_cursor();
        }
        self.send_mouse(dx, dy, 0x0001, false); // MOUSEEVENTF_MOVE
    }

    pub fn mouse_down(&mut self, button: &str) {
        let flag = match button {
            "left" => 0x0002,
            "right" => 0x0008,
            "middle" => 0x0020,
            _ => 0,
        };
        if flag != 0 {
            // A button goes down where the pointer is, so the pointer has to be
            // inside the game first.
            if self.may_confine() {
                self.confine_cursor();
            }
            self.send_mouse(0, 0, flag, false);
        }
    }

    pub fn mouse_up(&mut self, button: &str) {
        let flag = match button {
            "left" => 0x0004,
            "right" => 0x0010,
            "middle" => 0x0040,
            _ => 0,
        };
        if flag != 0 {
            // Forced for the same reason as a key release: never leave a button
            // down because the window lost focus mid-click.
            self.send_mouse(0, 0, flag, true);
        }
    }

    pub fn click(&mut self, button: &str, seconds: f32) {
        let vk = match button {
            "left" => Some(0x01),
            "right" => Some(0x02),
            "middle" => Some(0x04),
            _ => None,
        };
        if let Some(vk) = vk {
            self.transient_vks.insert(vk);
        }
        self.mouse_down(button);
        std::thread::sleep(std::time::Duration::from_secs_f32(seconds.max(0.0)));
        self.mouse_up(button);
        if let Some(vk) = vk {
            self.transient_vks.remove(&vk);
        }
    }

    pub fn release_all(&mut self, held_vks: &[u32]) {
        let held: Vec<u32> = held_vks.to_vec();
        for vk in held {
            self.release_vk(vk);
        }
    }
}

impl Drop for Injector {
    /// Never leave the user's cursor trapped in a window this process has
    /// stopped playing.
    ///
    /// This is the same promise the key releases make, for the other piece of
    /// global state the bot touches: keys are undone by the session's own exit
    /// path, but the cursor clip is desktop-wide, so it is given back on the way
    /// out even when the run ends by panicking.
    fn drop(&mut self) {
        self.release_cursor_clip();
    }
}


/// Executable name owning `pid`, or an empty string when it cannot be read.
pub fn get_process_name(pid: u32) -> String {
    if pid == 0 {
        return String::new();
    }
    let handle = match unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) } {
        Ok(handle) => handle,
        Err(_) => return String::new(),
    };
    let mut buffer = [0u16; MAX_PATH as usize];
    let mut size = buffer.len() as u32;
    let name = unsafe {
        match QueryFullProcessImageNameW(
            handle,
            Default::default(),
            PWSTR(buffer.as_mut_ptr()),
            &mut size,
        ) {
            Ok(()) => {
                let text = String::from_utf16_lossy(&buffer[..size as usize]);
                text.rsplit(['\\', '/']).next().unwrap_or("").to_string()
            }
            Err(_) => String::new(),
        }
    };
    let _ = unsafe { CloseHandle(handle) };
    name
}

pub fn client_size(handle: Handle) -> (i32, i32) {
    let mut rect = RECT::default();
    if unsafe { GetClientRect(HWND(handle as *mut _), &mut rect) }.is_err() {
        return (0, 0);
    }
    (rect.right, rect.bottom)
}

/// The window's client area, in the screen coordinates a cursor clip is
/// measured in.
///
/// The client area rather than the whole frame: the title bar and the borders
/// belong to Windows, so confining the cursor to them would only move the
/// problem - a click on the caption is a click on the window's chrome, not on
/// the game. `None` when the handle has no real area, which the caller has to
/// tolerate rather than clip to something empty.
///
/// These are the coordinates *this process* sees, which on a scaled display are
/// a 96-dpi virtual space rather than physical pixels. That is the right space:
/// `ClipCursor` takes its rectangle in the calling thread's space too, and
/// converts. Measured, not assumed - a clip set from an unaware thread reads
/// back through a per-monitor-aware thread scaled by the display's factor (1.25
/// on the machine this was written on), which is the conversion happening.
pub fn client_rect_on_screen(handle: Handle) -> Option<RECT> {
    use windows::Win32::Graphics::Gdi::ClientToScreen;
    if handle == 0 {
        return None;
    }
    let hwnd = HWND(handle as *mut _);
    let mut rect = RECT::default();
    if unsafe { GetClientRect(hwnd, &mut rect) }.is_err() {
        return None;
    }
    if rect.right <= rect.left || rect.bottom <= rect.top {
        return None;
    }
    let mut top_left = POINT {
        x: rect.left,
        y: rect.top,
    };
    let mut bottom_right = POINT {
        x: rect.right,
        y: rect.bottom,
    };
    unsafe {
        // Both corners: the client area need not start at the screen origin and
        // need not be on the primary monitor.
        let _ = ClientToScreen(hwnd, &mut top_left);
        let _ = ClientToScreen(hwnd, &mut bottom_right);
    }
    Some(RECT {
        left: top_left.x,
        top: top_left.y,
        right: bottom_right.x,
        bottom: bottom_right.y,
    })
}

/// Let the cursor out of whatever window holds it, claimed by this process or
/// not.
///
/// The injector releases its own clip on the way out, but two exits never unwind
/// and so never run that: the second Ctrl+C, which exits immediately by design,
/// and `diagnostics::release_all_keys`, whose whole purpose is to clean up after
/// a run that died. A clip is desktop-wide state and nothing else in this
/// process will undo it, so those paths call this instead.
pub fn free_cursor_clip() {
    let _ = unsafe { ClipCursor(None) };
}

pub fn is_minimized(handle: Handle) -> bool {
    unsafe { IsIconic(HWND(handle as *mut _)).as_bool() }
}

pub fn is_window(handle: Handle) -> bool {
    handle != 0 && unsafe { IsWindow(Some(HWND(handle as *mut _))).as_bool() }
}

pub fn foreground_window() -> Handle {
    unsafe { GetForegroundWindow().0 as isize }
}

/// The console window this process writes to, or 0.
///
/// A classic console answers here; Windows Terminal and an IDE terminal do not,
/// because ConPTY hands the process a hidden pseudo-window instead of the window
/// the user can see. `own_process_ids` covers those.
pub fn console_window() -> Handle {
    unsafe { GetConsoleWindow().0 as isize }
}

/// This process plus every ancestor: the shell, the terminal, the IDE.
///
/// Walking the parent chain is what makes this work where `GetConsoleWindow`
/// does not: running the bot from Windows Terminal gives a hidden ConPTY window,
/// not the visible terminal, so the only reliable link to the window the user is
/// looking at is that Windows Terminal started the shell that started the bot.
pub fn own_process_ids() -> HashSet<u32> {
    let mut ids = HashSet::new();
    let own = unsafe { GetCurrentProcessId() };
    ids.insert(own);

    let parents = process_parents();
    let mut pid = own;
    // Depth cap: never loop forever, whatever the snapshot says.
    for _ in 0..16 {
        let parent = parents.get(&pid).copied().unwrap_or(0);
        if parent == 0 || ids.contains(&parent) {
            break;
        }
        ids.insert(parent);
        pid = parent;
    }
    ids
}

fn process_parents() -> HashMap<u32, u32> {
    let mut parents = HashMap::new();
    let snapshot = match unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) } {
        Ok(snapshot) => snapshot,
        Err(_) => return parents,
    };
    let mut entry = PROCESSENTRY32W {
        dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32,
        ..Default::default()
    };
    let mut more = unsafe { Process32FirstW(snapshot, &mut entry) }.is_ok();
    while more {
        parents.insert(entry.th32ProcessID, entry.th32ParentProcessID);
        more = unsafe { Process32NextW(snapshot, &mut entry) }.is_ok();
    }
    let _ = unsafe { CloseHandle(snapshot) };
    parents
}

/// Does this window belong to the bot, or to whatever launched it?
pub fn window_is_own_process(handle: Handle) -> bool {
    if handle == 0 {
        return false;
    }
    if handle == console_window() {
        return true;
    }
    let pid = get_window_pid(handle);
    pid > 0 && own_process_ids().contains(&pid)
}

/// Every visible, titled, large-enough top-level window, largest first.
pub fn enumerate_windows(min_area: i32) -> Vec<WindowInfo> {
    let own = own_process_ids();
    let console = console_window();
    // The callback cannot see the process table or this stack frame, so what it
    // needs travels in the LPARAM: the list it appends to, and the area floor.
    let mut state = EnumState {
        found: Vec::new(),
        min_area,
    };
    let pointer = &mut state as *mut EnumState as isize;
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(pointer));
    }
    let mut found = state.found;
    for window in found.iter_mut() {
        window.own = window.pid > 0 && own.contains(&window.pid);
        window.console = console != 0 && window.handle == console;
    }
    found.sort_by(|a, b| {
        let area = |w: &WindowInfo| w.rect.2.saturating_mul(w.rect.3);
        area(b).cmp(&area(a))
    });
    found
}

struct EnumState {
    found: Vec<WindowInfo>,
    min_area: i32,
}

unsafe extern "system" fn visit(hwnd: HWND, lparam: LPARAM) -> windows::core::BOOL {
    use windows::core::BOOL;
    let state = unsafe { &mut *(lparam.0 as *mut EnumState) };
    let raw = hwnd.0 as isize;
    if let Some(entry) = entry_for(hwnd, raw) {
        let area = entry.rect.2.saturating_mul(entry.rect.3);
        if area >= state.min_area {
            state.found.push(entry);
        }
    }
    BOOL(1)
}

/// One window in the shape [`enumerate_windows`] produces, by handle.
pub fn window_entry(handle: Handle) -> Option<WindowInfo> {
    if !is_window(handle) {
        return None;
    }
    entry_for(HWND(handle as *mut _), handle)
}

fn entry_for(hwnd: HWND, raw: Handle) -> Option<WindowInfo> {
    if !unsafe { IsWindowVisible(hwnd).as_bool() } {
        return None;
    }
    let title = window_text(hwnd);
    if title.trim().is_empty() {
        return None;
    }
    let mut rect = RECT::default();
    if unsafe { GetWindowRect(hwnd, &mut rect) }.is_err() {
        return None;
    }
    let pid = get_window_pid(raw);
    Some(WindowInfo {
        handle: raw,
        title,
        pid,
        process: get_process_name(pid),
        rect: (
            rect.left,
            rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top,
        ),
        client: client_size(raw),
        minimized: is_minimized(raw),
        own: false,
        console: false,
    })
}

// =============================================================================
// Global hotkeys
// =============================================================================

const WM_HOTKEY: u32 = 0x0312;

/// The start/pause, checkpoint and quit keys, registered globally.
///
/// `RegisterHotKey` with a null window posts `WM_HOTKEY` to the *calling
/// thread's* queue, so registration and the message loop have to be the same
/// thread. That is why this owns a thread rather than borrowing one: the
/// alternative is registering on the main thread and never pumping it, which
/// looks exactly like hotkeys that do not work.
pub struct Hotkeys {
    specs: Vec<(HotkeyAction, String)>,
    actions: Arc<Mutex<Vec<HotkeyAction>>>,
    messages: Arc<Mutex<Vec<String>>>,
    registered: Arc<Mutex<Vec<(HotkeyAction, String)>>>,
    thread_id: Arc<AtomicU32>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Hotkeys {
    pub fn new(pause: &str, save: &str, quit: &str) -> Self {
        Self {
            specs: vec![
                (HotkeyAction::Pause, pause.to_string()),
                (HotkeyAction::Save, save.to_string()),
                (HotkeyAction::Quit, quit.to_string()),
            ],
            actions: Arc::new(Mutex::new(Vec::new())),
            messages: Arc::new(Mutex::new(Vec::new())),
            registered: Arc::new(Mutex::new(Vec::new())),
            thread_id: Arc::new(AtomicU32::new(0)),
            thread: None,
        }
    }

    /// Register everything and start listening. Returns whether any of them
    /// registered, after waiting up to three seconds for the thread to say so.
    pub fn start(&mut self) -> bool {
        let specs = self.specs.clone();
        let actions = Arc::clone(&self.actions);
        let messages = Arc::clone(&self.messages);
        let registered = Arc::clone(&self.registered);
        let thread_id = Arc::clone(&self.thread_id);
        let (ready_sender, ready_receiver) = std::sync::mpsc::channel::<()>();

        let handle = std::thread::Builder::new()
            .name("hotkeys".to_string())
            .spawn(move || {
                hotkey_loop(specs, actions, messages, registered, thread_id, ready_sender);
            });
        match handle {
            Ok(handle) => self.thread = Some(handle),
            Err(error) => {
                self.messages
                    .lock()
                    .map(|mut queue| queue.push(format!("Hotkeys unavailable: {error}")))
                    .ok();
                return false;
            }
        }
        // Bounded wait only: a thread that never reports must not hang a run.
        let _ = ready_receiver.recv_timeout(std::time::Duration::from_secs(3));
        self.available()
    }

    pub fn available(&self) -> bool {
        self.registered
            .lock()
            .map(|list| !list.is_empty())
            .unwrap_or(false)
    }

    /// Hotkeys pressed since the last call, oldest first.
    pub fn poll(&mut self) -> Vec<HotkeyAction> {
        self.actions
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default()
    }

    pub fn drain_messages(&mut self) -> Vec<String> {
        self.messages
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default()
    }

    pub fn describe(&self) -> String {
        let registered = self.registered.lock().map(|list| list.clone()).unwrap_or_default();
        if registered.is_empty() {
            return "none".to_string();
        }
        ["pause", "save", "quit"]
            .iter()
            .filter_map(|name| {
                registered
                    .iter()
                    .find(|(action, _)| action.name() == *name)
                    .map(|(action, spec)| format!("{}={}", action.name().to_uppercase(), spec.to_uppercase()))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Ask the listener to stop and unregister.
    pub fn stop(&mut self) {
        use windows::Win32::UI::WindowsAndMessaging::PostThreadMessageW;
        let id = self.thread_id.load(Ordering::SeqCst);
        if id != 0 {
            unsafe {
                let _ = PostThreadMessageW(id, WM_QUIT, WPARAM(0), LPARAM(0));
            }
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for Hotkeys {
    fn drop(&mut self) {
        self.stop();
    }
}

const WM_QUIT: u32 = 0x0012;

fn hotkey_loop(
    specs: Vec<(HotkeyAction, String)>,
    actions: Arc<Mutex<Vec<HotkeyAction>>>,
    messages: Arc<Mutex<Vec<String>>>,
    registered: Arc<Mutex<Vec<(HotkeyAction, String)>>>,
    thread_id: Arc<AtomicU32>,
    ready: std::sync::mpsc::Sender<()>,
) {
    use windows::Win32::UI::Input::KeyboardAndMouse::{HOT_KEY_MODIFIERS, RegisterHotKey, UnregisterHotKey};
    use windows::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG};

    thread_id.store(unsafe { GetCurrentThreadId() }, Ordering::SeqCst);
    let mut ids: Vec<(i32, HotkeyAction)> = Vec::new();
    // Identifiers start above the range a shell uses, and are consumed even by a
    // registration that fails, exactly as the Python's counter is.
    let mut next_id: i32 = 0xB001;

    for (action, spec) in &specs {
        if spec.is_empty() {
            continue;
        }
        let (modifiers, vk) = match crate::keys::parse_hotkey(spec) {
            Ok(parsed) => parsed,
            Err(message) => {
                messages.lock().map(|mut queue| queue.push(message)).ok();
                continue;
            }
        };
        let id = next_id;
        next_id += 1;
        let result = unsafe {
            RegisterHotKey(
                None,
                id,
                HOT_KEY_MODIFIERS(modifiers | crate::keys::MOD_NOREPEAT),
                vk,
            )
        };
        match result {
            Ok(()) => {
                registered
                    .lock()
                    .map(|mut list| list.push((*action, spec.clone())))
                    .ok();
                ids.push((id, *action));
            }
            Err(_) => {
                messages
                    .lock()
                    .map(|mut queue| {
                        queue.push(format!(
                            "Could not register {} for '{}' - another program may already use \
                             it.",
                            spec.to_uppercase(),
                            action.name()
                        ))
                    })
                    .ok();
            }
        }
    }

    let _ = ready.send(());
    if ids.is_empty() {
        return;
    }

    let mut message = MSG::default();
    loop {
        let status = unsafe { GetMessageW(&mut message, None, 0, 0) };
        if status.0 <= 0 {
            // Zero is WM_QUIT; -1 is an error. Both end the listener.
            break;
        }
        if message.message == WM_HOTKEY {
            let id = message.wParam.0 as i32;
            if let Some((_, action)) = ids.iter().find(|(registered_id, _)| *registered_id == id)
                && let Ok(mut queue) = actions.lock()
            {
                queue.push(*action);
            }
        }
    }

    for (id, _) in ids {
        unsafe {
            let _ = UnregisterHotKey(None, id);
        }
    }
    registered.lock().map(|mut list| list.clear()).ok();
}

// =============================================================================
// The calibration recorder
// =============================================================================

const WH_KEYBOARD_LL: i32 = 13;
const WH_MOUSE_LL: i32 = 14;
const WM_KEYDOWN: u32 = 0x0100;
const WM_KEYUP: u32 = 0x0101;
const WM_SYSKEYDOWN: u32 = 0x0104;
const WM_SYSKEYUP: u32 = 0x0105;
const WM_MOUSEMOVE: u32 = 0x0200;
/// Set in `KBDLLHOOKSTRUCT.flags` when the event came from software.
const LLKHF_INJECTED: u32 = 0x10;
/// The same bit in `MSLLHOOKSTRUCT.flags`.
const LLMHF_INJECTED: u32 = 0x01;

/// Keys the recorder refuses to learn and the bot refuses to press: the game's
/// own debug overlays, the Windows keys, Alt, Tab and Escape.
const BLOCKED_VKS: [u32; 8] = [0x70, 0x72, 0x5B, 0x5C, 0x5D, 0x12, 0x09, 0x1B];

/// A keyboard hook event, read out of a `KBDLLHOOKSTRUCT`.
///
/// The layout is `vkCode` at 0, `scanCode` at 4, `flags` at 8, `time` at 12,
/// `dwExtraInfo` at 16. **`flags` is at offset 8, not 4.** The Python's
/// recorder reads index 1 - the scan code - and treats bit 4 of *that* as the
/// injected flag, which is why its calibration silently drops every key whose
/// scan code has bit 4 set: W (0x11), A (0x1E), S (0x1F), Q, E, R, T, LCTRL.
/// See `rust/NOTES.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyboardEvent {
    pub vk: u32,
    pub scan_code: u32,
    pub flags: u32,
}

impl KeyboardEvent {
    pub fn injected(&self) -> bool {
        self.flags & LLKHF_INJECTED != 0
    }

    pub fn down(&self, wparam: u32) -> bool {
        match wparam {
            WM_KEYDOWN | WM_SYSKEYDOWN => true,
            WM_KEYUP | WM_SYSKEYUP => false,
            // Neither a press nor a release: not a transition the recorder acts
            // on.
            _ => false,
        }
    }
}

/// Read a `KBDLLHOOKSTRUCT` out of the bytes a hook was handed.
pub fn parse_keyboard_event(bytes: &[u8]) -> Option<KeyboardEvent> {
    if bytes.len() < 12 {
        return None;
    }
    let read = |offset: usize| -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    Some(KeyboardEvent {
        vk: read(0),
        scan_code: read(4),
        flags: read(8),
    })
}

/// A mouse hook event, read out of an `MSLLHOOKSTRUCT`.
///
/// The layout is `pt.x` at 0, `pt.y` at 4, `mouseData` at 8, `flags` at 12. The
/// Python reads the first two as a `POINT` correctly, and never reads `flags` at
/// all - so during a run's own calibration it records the bot's injected mouse
/// motion as the user's. This reads it, and the recorder skips injected events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MouseEvent {
    pub x: i32,
    pub y: i32,
    pub flags: u32,
}

impl MouseEvent {
    pub fn injected(&self) -> bool {
        self.flags & LLMHF_INJECTED != 0
    }
}

pub fn parse_mouse_event(bytes: &[u8]) -> Option<MouseEvent> {
    if bytes.len() < 16 {
        return None;
    }
    let read_u32 = |offset: usize| -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    };
    Some(MouseEvent {
        x: read_u32(0) as i32,
        y: read_u32(4) as i32,
        flags: read_u32(12),
    })
}

/// The recorder's state, which the hook callbacks reach through a thread-local.
///
/// A low-level hook callback gets no user parameter, and it runs on the thread
/// that installed the hook - so a thread-local is the only way to hand it state
/// without a global. Everything in here is therefore touched only from that
/// thread: `start`, `pump` and `stop` all have to be called from it.
#[derive(Default)]
pub struct RecorderState {
    pub toggle_vk: u32,
    pub target_handle: Handle,
    pub only_when_focused: bool,
    pub recording: bool,
    pub toggles: u64,
    pub events: u64,
    pub focus_ignored: u64,
    pub pressed: HashMap<String, std::time::Instant>,
    pub stats: BTreeMap<String, crate::keys::KeyStats>,
    pub mouse_pixels: Vec<f64>,
    last_toggle: Option<std::time::Instant>,
    mouse_pos: Option<(i32, i32)>,
}

impl RecorderState {
    fn wants_input(&self) -> bool {
        if !self.only_when_focused || self.target_handle == 0 {
            return true;
        }
        foreground_window() == self.target_handle
    }

    fn note_press(&mut self, name: &str) {
        // Auto-repeat arrives as repeated key-downs; only the first counts, so
        // the recorded hold spans every repeat.
        if self.pressed.contains_key(name) {
            return;
        }
        self.pressed
            .insert(name.to_string(), std::time::Instant::now());
        self.events += 1;
    }

    fn close_key(&mut self, name: &str) {
        let Some(started) = self.pressed.remove(name) else {
            return;
        };
        let held = started.elapsed().as_secs_f64().max(0.0);
        let entry = self.stats.entry(name.to_string()).or_default();
        entry.count += 1;
        entry.total_held += held;
        entry.longest_hold = entry.longest_hold.max(held);
    }

    pub fn toggle(&mut self) -> bool {
        self.toggles += 1;
        self.last_toggle = Some(std::time::Instant::now());
        self.recording = !self.recording;
        if self.recording {
            // The first move after starting must not be measured from wherever
            // the cursor was before.
            self.mouse_pos = None;
        } else {
            let names: Vec<String> = self.pressed.keys().cloned().collect();
            for name in names {
                self.close_key(&name);
            }
        }
        self.recording
    }

    pub fn snapshot(&self, target: Option<&WindowInfo>) -> crate::keys::Recording {
        crate::keys::Recording {
            stats: self.stats.clone(),
            mouse_pixels: self.mouse_pixels.clone(),
            events: self.events,
            target_handle: self.target_handle.checked_abs().map(|h| h as usize),
            target_title: target.map(|w| w.title.clone()).unwrap_or_default(),
            target_client: target.map(|w| w.client).unwrap_or_default(),
        }
    }

    fn note_keyboard(&mut self, event: &KeyboardEvent, wparam: u32) {
        let down = event.down(wparam);
        if event.vk == self.toggle_vk && down && !event.injected() {
            // A 0.35 s debounce, measured against the last *accepted* toggle.
            let fresh = self
                .last_toggle
                .map(|last| last.elapsed().as_secs_f64() > 0.35)
                .unwrap_or(true);
            if fresh {
                self.toggle();
            }
            return;
        }
        if !self.recording || event.injected() {
            return;
        }
        if !self.wants_input() {
            self.focus_ignored += 1;
            return;
        }
        if BLOCKED_VKS.contains(&event.vk) {
            return;
        }
        let name = crate::keys::key_name(event.vk, false);
        if down {
            self.note_press(&name);
        } else {
            self.close_key(&name);
        }
    }

    fn note_mouse(&mut self, event: &MouseEvent, wparam: u32) {
        if !self.recording || event.injected() {
            return;
        }
        if !self.wants_input() {
            self.focus_ignored += 1;
            return;
        }
        if wparam == WM_MOUSEMOVE {
            if let Some((last_x, last_y)) = self.mouse_pos {
                let dx = (event.x - last_x) as f64;
                let dy = (event.y - last_y) as f64;
                if dx != 0.0 || dy != 0.0 {
                    self.mouse_pixels
                        .push((dx * dx + dy * dy).sqrt());
                    self.events += 1;
                }
            }
            // Unconditionally: a zero-delta move still advances the reference.
            self.mouse_pos = Some((event.x, event.y));
            return;
        }
        let button = match wparam {
            0x0201 => Some(("MOUSE_LEFT", true)),
            0x0202 => Some(("MOUSE_LEFT", false)),
            0x0204 => Some(("MOUSE_RIGHT", true)),
            0x0205 => Some(("MOUSE_RIGHT", false)),
            0x0207 => Some(("MOUSE_MIDDLE", true)),
            0x0208 => Some(("MOUSE_MIDDLE", false)),
            _ => None,
        };
        if let Some((name, down)) = button {
            if down {
                self.note_press(name);
            } else {
                self.close_key(name);
            }
        }
    }
}

thread_local! {
    static RECORDER: std::cell::RefCell<RecorderState> =
        std::cell::RefCell::new(RecorderState::default());
}

/// Run something against the recorder state, if it is not already borrowed.
///
/// A hook callback that re-entered the state would otherwise panic, and a panic
/// across an FFI boundary is undefined behaviour.
fn with_state<R>(body: impl FnOnce(&mut RecorderState) -> R) -> Option<R> {
    RECORDER.with(|cell| cell.try_borrow_mut().ok().map(|mut state| body(&mut state)))
}

/// Reads the keyboard from the whole session, without stealing a keystroke.
pub struct KeyRecorder {
    keyboard_hook: HHOOK,
    mouse_hook: HHOOK,
    pub running: bool,
}

impl KeyRecorder {
    /// Install both hooks. They have to be installed, pumped and removed on the
    /// same thread.
    pub fn start(toggle_vk: u32, target: Handle, only_when_focused: bool) -> Self {
        use windows::Win32::UI::WindowsAndMessaging::{
            SetWindowsHookExW, WINDOWS_HOOK_ID,
        };
        with_state(|state| {
            state.toggle_vk = toggle_vk;
            state.target_handle = target;
            state.only_when_focused = only_when_focused;
        });
        let keyboard_hook = unsafe {
            SetWindowsHookExW(
                WINDOWS_HOOK_ID(WH_KEYBOARD_LL),
                Some(keyboard_proc),
                None,
                0,
            )
        }
        .unwrap_or_default();
        let mouse_hook = unsafe {
            SetWindowsHookExW(WINDOWS_HOOK_ID(WH_MOUSE_LL), Some(mouse_proc), None, 0)
        }
        .unwrap_or_default();
        Self {
            keyboard_hook,
            mouse_hook,
            running: true,
        }
    }

    pub fn installed(&self) -> bool {
        !self.keyboard_hook.is_invalid()
    }

    /// Deliver any pending messages, which is what lets Windows run the hooks.
    pub fn pump(&mut self) {
        use windows::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, MSG, PM_REMOVE, PeekMessageW, TranslateMessage,
        };
        let mut message = MSG::default();
        while unsafe { PeekMessageW(&mut message, None, 0, 0, PM_REMOVE) }.as_bool() {
            unsafe {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }

    pub fn recording(&self) -> bool {
        with_state(|state| state.recording).unwrap_or(false)
    }

    pub fn toggles(&self) -> u64 {
        with_state(|state| state.toggles).unwrap_or(0)
    }

    pub fn events(&self) -> u64 {
        with_state(|state| state.events).unwrap_or(0)
    }

    /// Stop recording and let go of both hooks.
    pub fn stop(&mut self) {
        // Flush whatever is still held down, so a key the user is holding when
        // they stop still counts as a hold.
        let names = with_state(|state| {
            state.recording = false;
            state.pressed.keys().cloned().collect::<Vec<String>>()
        })
        .unwrap_or_default();
        for name in names {
            with_state(|state| state.close_key(&name));
        }
        self.running = false;
        use windows::Win32::UI::WindowsAndMessaging::UnhookWindowsHookEx;
        unsafe {
            if !self.keyboard_hook.is_invalid() {
                let _ = UnhookWindowsHookEx(self.keyboard_hook);
            }
            if !self.mouse_hook.is_invalid() {
                let _ = UnhookWindowsHookEx(self.mouse_hook);
            }
        }
        self.keyboard_hook = HHOOK::default();
        self.mouse_hook = HHOOK::default();
    }

    pub fn snapshot(&self, target: Option<&WindowInfo>) -> crate::keys::Recording {
        with_state(|state| state.snapshot(target))
            .unwrap_or_else(crate::keys::Recording::default)
    }
}

impl Drop for KeyRecorder {
    fn drop(&mut self) {
        if self.running {
            self.stop();
        }
    }
}

unsafe extern "system" fn keyboard_proc(
    code: i32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    // Negative codes mean "pass it on without processing".
    if code >= 0 && lparam.0 != 0 {
        let bytes = unsafe { std::slice::from_raw_parts(lparam.0 as *const u8, 24) };
        if let Some(event) = parse_keyboard_event(bytes) {
            with_state(|state| state.note_keyboard(&event, wparam.0 as u32));
        }
    }
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

unsafe extern "system" fn mouse_proc(
    code: i32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> windows::Win32::Foundation::LRESULT {
    if code >= 0 && lparam.0 != 0 {
        let bytes = unsafe { std::slice::from_raw_parts(lparam.0 as *const u8, 16) };
        if let Some(event) = parse_mouse_event(bytes) {
            with_state(|state| state.note_mouse(&event, wparam.0 as u32));
        }
    }
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::MIN_AREA;

    #[test]
    fn a_terminal_is_never_a_likely_game() {
        assert!(is_non_game_process("pwsh.exe"));
        assert!(is_non_game_process("chrome.exe"));
        assert!(!is_non_game_process("game.exe"));
    }

    #[test]
    fn this_process_and_its_ancestors_are_marked_own() {
        let own = own_process_ids();
        let me = unsafe { GetCurrentProcessId() };
        assert!(own.contains(&me), "the process itself must be in its own set");
    }

    #[test]
    fn enumeration_finds_the_windows_it_should() {
        // A real call to EnumWindows: every entry must carry a handle and a
        // title, and the list must be sorted by area, largest first.
        let windows = enumerate_windows(MIN_AREA);
        for window in &windows {
            assert_ne!(window.handle, 0);
            assert!(!window.title.trim().is_empty());
        }
        for pair in windows.windows(2) {
            let area = |w: &WindowInfo| w.rect.2.saturating_mul(w.rect.3);
            assert!(area(&pair[0]) >= area(&pair[1]));
        }
    }

    /// The load-bearing safety property: a decision made while the game is not
    /// in front must not reach the keyboard.
    ///
    /// This test sends nothing, and that is the point. The target handle is one
    /// that is not the foreground window, so every unforced call is refused
    /// before `SendInput`; and the forced paths (a key release, used so a key
    /// can never be left down) are exercised with injection disabled, which is
    /// the switch calibration uses to shout if anything tries.
    #[test]
    fn nothing_is_sent_to_a_window_that_is_not_in_front() {
        use crate::config::FocusPolicy;
        // Handle 1 is not the foreground window, so `focused()` is false and the
        // unforced paths are refused.
        let mut injector = Injector::new(1, FocusPolicy::Once, true);
        assert!(!injector.focused());
        injector.begin_action();
        injector.press_vk(0x57);
        injector.mouse_move(10, 10);
        injector.mouse_down("left");
        injector.tap_vk(0x20, 0.0);
        assert_eq!(injector.events_sent, 0, "nothing may leave this process");
        assert_eq!(injector.keys_sent, 0);
        assert_eq!(injector.mouse_sent, 0);
        assert_eq!(injector.failures, 0);
        assert!(injector.skipped_unfocused >= 4, "each attempt is counted");

        // With injection disabled, even a forced release is refused - which is
        // what makes it safe to press keys during calibration.
        let mut disabled = Injector::new(1, FocusPolicy::Once, true);
        disabled.enabled = false;
        disabled.release_vk(0x57);
        assert_eq!(disabled.events_sent, 0);
        assert_eq!(disabled.blocked, 1);

        // A suspended injector is silent: the user has the controls, and that is
        // expected rather than a bug worth reporting.
        let mut suspended = Injector::new(1, FocusPolicy::Once, true);
        suspended.suspended = true;
        suspended.release_vk(0x57);
        assert_eq!(suspended.events_sent, 0);
        assert_eq!(suspended.blocked, 0, "a pause is not a bug");
    }

    /// The clip rectangle Windows is holding, as plain numbers - so a test can
    /// compare two of them without caring how `RECT` compares.
    fn cursor_clip() -> Option<(i32, i32, i32, i32)> {
        use windows::Win32::UI::WindowsAndMessaging::GetClipCursor;
        let mut rect = RECT::default();
        unsafe { GetClipCursor(&mut rect) }.ok()?;
        Some((rect.left, rect.top, rect.right, rect.bottom))
    }

    /// The second load-bearing safety property: the cursor belongs to the user,
    /// so an action that is refused must not confine it, and one that never
    /// claimed it must not release it either.
    #[test]
    fn a_refused_action_never_touches_the_cursor_clip() {
        use crate::config::FocusPolicy;
        let before = cursor_clip();
        // Handle 1 is not the foreground window, so nothing is confined.
        let mut injector = Injector::new(1, FocusPolicy::Once, true);
        injector.begin_action();
        injector.mouse_move(40, 40);
        injector.tap_vk(0x01, 0.0); // a mouse-button tap, the click path
        injector.mouse_down("left");
        assert_eq!(injector.mouse_sent, 0, "every move was refused");
        injector.end_action();
        // `end_action` runs the release, and it must be a no-op here: an
        // unconditional `ClipCursor(NULL)` would free a clip the *game* set.
        assert_eq!(cursor_clip(), before, "the desktop clip must be untouched");

        // And with confinement switched off, the injector never claims it.
        let mut off = Injector::new(1, FocusPolicy::Once, false);
        off.begin_action();
        off.mouse_move(40, 40);
        off.end_action();
        assert!(!off.confined);
        assert_eq!(off.clip_failures, 0, "off is not a failure");
        assert_eq!(cursor_clip(), before);
    }

    #[test]
    fn a_window_that_is_not_a_window_has_no_client_rectangle() {
        // The clip rectangle is built from `GetClientRect` + `ClientToScreen`,
        // and a bogus handle must produce nothing rather than an empty rect at
        // the screen origin - which would pin the cursor to the top-left pixel.
        assert!(client_rect_on_screen(0).is_none());
        assert!(client_rect_on_screen(1).is_none());
    }

    #[test]
    fn the_extended_keys_are_the_ones_that_need_the_flag() {
        // Arrows, the navigation cluster, right-hand modifiers and the Windows
        // keys: without the flag they arrive as their numpad equivalents.
        for vk in [0xA1u32, 0xA3, 0xA5, 0x25, 0x26, 0x27, 0x28, 0x5B, 0x5C] {
            assert!(EXTENDED_VKS.contains(&vk), "vk {vk:#X} should be extended");
        }
        // And the letters must not be.
        for vk in [0x57u32, 0x41, 0x20, 0x0D] {
            assert!(!EXTENDED_VKS.contains(&vk), "vk {vk:#X} is not extended");
        }
    }

    /// The regression test for the calibration bug: `flags` is at offset 8, and
    /// W's scan code has exactly the bit the Python reads as "injected".
    #[test]
    fn the_keyboard_hook_struct_is_read_at_the_right_offsets() {
        // KBDLLHOOKSTRUCT: vkCode(0), scanCode(4), flags(8), time(12), extra(16).
        let mut bytes = [0u8; 24];
        bytes[0..4].copy_from_slice(&0x57u32.to_le_bytes()); // W
        bytes[4..8].copy_from_slice(&0x11u32.to_le_bytes()); // W's scan code
        let event = parse_keyboard_event(&bytes).expect("a 24-byte struct");
        assert_eq!(event.vk, 0x57);
        assert_eq!(event.scan_code, 0x11);
        assert_eq!(event.flags, 0);
        assert!(
            !event.injected(),
            "scan code 0x11 has bit 4 set; reading it as `flags` would drop W, A and S"
        );

        // The bit that actually means "injected" is in the flags field.
        bytes[8..12].copy_from_slice(&LLKHF_INJECTED.to_le_bytes());
        assert!(parse_keyboard_event(&bytes).unwrap().injected());

        // Every key the Python's aliasing loses, by scan code.
        for (name, scan) in [("W", 0x11u32), ("A", 0x1E), ("S", 0x1F), ("Q", 0x10)] {
            let mut bytes = [0u8; 24];
            bytes[4..8].copy_from_slice(&scan.to_le_bytes());
            let event = parse_keyboard_event(&bytes).unwrap();
            assert!(
                !event.injected(),
                "{name} (scan {scan:#X}) must still be recordable"
            );
        }
        // And D, which the Python does record, for contrast.
        let mut bytes = [0u8; 24];
        bytes[4..8].copy_from_slice(&0x20u32.to_le_bytes());
        assert!(!parse_keyboard_event(&bytes).unwrap().injected());
    }

    #[test]
    fn the_mouse_hook_struct_is_read_at_the_right_offsets() {
        // MSLLHOOKSTRUCT: pt.x(0), pt.y(4), mouseData(8), flags(12), ...
        let mut bytes = [0u8; 32];
        bytes[0..4].copy_from_slice(&(-12i32).to_le_bytes());
        bytes[4..8].copy_from_slice(&34i32.to_le_bytes());
        let event = parse_mouse_event(&bytes).expect("a 32-byte struct");
        assert_eq!((event.x, event.y), (-12, 34));
        assert!(!event.injected());
        bytes[12..16].copy_from_slice(&LLMHF_INJECTED.to_le_bytes());
        assert!(parse_mouse_event(&bytes).unwrap().injected());
    }

    #[test]
    fn the_recorder_measures_a_hold_and_collapses_auto_repeat() {
        let mut state = RecorderState {
            // Not focus-gated, so the test does not touch the window system.
            only_when_focused: false,
            recording: true,
            ..Default::default()
        };
        state.note_press("W");
        state.note_press("W"); // auto-repeat
        assert_eq!(state.events, 1, "a repeat is not a second press");
        std::thread::sleep(std::time::Duration::from_millis(20));
        state.close_key("W");
        let stats = state.stats.get("W").copied().expect("W was recorded");
        assert_eq!(stats.count, 1);
        assert!(stats.total_held >= 0.02);
        // A key-up with no key-down is ignored rather than counted.
        state.close_key("W");
        assert_eq!(state.stats.get("W").unwrap().count, 1);
    }

    #[test]
    fn the_recorder_ignores_its_own_injected_input() {
        let mut state = RecorderState {
            only_when_focused: false,
            recording: true,
            ..Default::default()
        };
        // A key the bot itself sent must not appear as the user's.
        let injected = KeyboardEvent {
            vk: 0x57,
            scan_code: 0x11,
            flags: LLKHF_INJECTED,
        };
        state.note_keyboard(&injected, WM_KEYDOWN);
        assert!(state.stats.is_empty());
        assert_eq!(state.events, 0);

        // Neither must an injected mouse move - the half of the bug the Python
        // does not check at all. It is dropped *and* it does not become the
        // reference point, which is what the next assertion measures.
        let injected_move = MouseEvent {
            x: 10,
            y: 10,
            flags: LLMHF_INJECTED,
        };
        state.note_mouse(&injected_move, WM_MOUSEMOVE);
        assert!(state.mouse_pixels.is_empty());

        // The first real move sets the reference and records nothing.
        state.note_mouse(&MouseEvent { x: 20, y: 10, flags: 0 }, WM_MOUSEMOVE);
        assert!(state.mouse_pixels.is_empty());

        // The second measures ten pixels from *that* point. Had the injected
        // move been taken as the reference, this would read twenty.
        state.note_mouse(&MouseEvent { x: 30, y: 10, flags: 0 }, WM_MOUSEMOVE);
        assert_eq!(state.mouse_pixels, vec![10.0]);
    }

    #[test]
    fn the_toggle_key_starts_and_stops_the_recording_and_is_never_recorded() {
        let mut state = RecorderState {
            toggle_vk: 0x77,
            only_when_focused: false,
            ..Default::default()
        };
        let toggle = KeyboardEvent {
            vk: 0x77,
            scan_code: 0x42,
            flags: 0,
        };
        assert!(!state.recording);
        state.note_keyboard(&toggle, WM_KEYDOWN);
        assert!(state.recording, "the first press starts recording");
        assert!(!state.stats.contains_key("F8"), "the toggle key is not a key");

        // A second press within the debounce window is ignored.
        state.note_keyboard(&toggle, WM_KEYDOWN);
        assert!(state.recording, "the debounce swallows a double press");
        std::thread::sleep(std::time::Duration::from_millis(400));
        state.note_keyboard(&toggle, WM_KEYDOWN);
        assert!(!state.recording, "a later press stops it");
        assert_eq!(state.toggles, 2);
    }

    #[test]
    fn the_recorder_refuses_the_blocked_keys() {
        let mut state = RecorderState {
            only_when_focused: false,
            recording: true,
            ..Default::default()
        };
        for vk in BLOCKED_VKS {
            state.note_keyboard(
                &KeyboardEvent {
                    vk,
                    scan_code: 0,
                    flags: 0,
                },
                WM_KEYDOWN,
            );
        }
        assert!(
            state.stats.is_empty(),
            "F1, F3, the Windows keys, Alt, Tab, Esc"
        );
        assert_eq!(state.events, 0);
    }

    /// The whole path, from a raw `KBDLLHOOKSTRUCT` to a recorded hold.
    ///
    /// The hooks are really installed and the callback is called *directly*,
    /// with a synthetic struct. Calling it rather than injecting is the point:
    /// a synthetic `SendInput` would have to be delivered through the user's
    /// input queue, and a test has no business putting events in front of
    /// someone else's window. Everything after Windows' own dispatch is
    /// exercised here - the thread-local state, the offset parsing, the
    /// focus gate, the auto-repeat collapse and the hold accounting.
    #[test]
    fn a_hook_callback_records_a_key_it_is_handed() {
        let mut recorder = KeyRecorder::start(0x77, 0, false);
        assert!(recorder.installed(), "the hooks install on this thread");
        with_state(|state| state.recording = true);

        // W, whose scan code 0x11 is the one the Python's aliasing loses.
        let mut pressed = [0u8; 24];
        pressed[0..4].copy_from_slice(&0x57u32.to_le_bytes());
        pressed[4..8].copy_from_slice(&0x11u32.to_le_bytes());
        unsafe {
            keyboard_proc(
                0,
                WPARAM(WM_KEYDOWN as usize),
                LPARAM(pressed.as_ptr() as isize),
            );
        }
        assert_eq!(recorder.events(), 1, "W is recorded");

        std::thread::sleep(std::time::Duration::from_millis(20));
        unsafe {
            keyboard_proc(
                0,
                WPARAM(WM_KEYUP as usize),
                LPARAM(pressed.as_ptr() as isize),
            );
        }
        let stats = with_state(|state| state.stats.get("W").copied())
            .flatten()
            .expect("W has statistics");
        assert_eq!(stats.count, 1);
        assert!(stats.total_held >= 0.02, "{stats:?}");

        // A negative code means "do not process this event".
        unsafe {
            keyboard_proc(
                -1,
                WPARAM(WM_KEYDOWN as usize),
                LPARAM(pressed.as_ptr() as isize),
            );
        }
        assert_eq!(recorder.events(), 1);

        // And the toggle key, handed to the same callback, flips the state.
        let mut toggle = [0u8; 24];
        toggle[0..4].copy_from_slice(&0x77u32.to_le_bytes());
        unsafe {
            keyboard_proc(
                0,
                WPARAM(WM_KEYDOWN as usize),
                LPARAM(toggle.as_ptr() as isize),
            );
        }
        assert!(!recorder.recording(), "F8 stops the recording");
        assert_eq!(recorder.toggles(), 1);
        recorder.stop();
    }
}
