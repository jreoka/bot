//! The checks that tell a user what the bot sees and what it may press.
//! Ported from `botcore/diagnostics.py`.
//!
//! These exist because the failure modes of a bot that drives someone else's
//! window are all silent: a window that refuses to render, a game that pauses
//! while unfocused, a calibration that recorded nothing, a policy that cannot
//! press the one key that matters. Each check below measures one of those and
//! says which one it is, rather than leaving the user to watch a log.

use crate::capture::FrameStack;
use crate::config::Config;
use crate::keys::{ActionSpace, Keymap};
use crate::platform::{self, Handle, WindowInfo};

/// Watch a window for a few seconds and report what the capture actually yields.
pub fn check_capture(cfg: &Config, handle: Handle, seconds: f32) -> i32 {
    println!();
    println!("{}", "-".repeat(74));
    println!("  CAPTURE CHECK");
    println!("{}", "-".repeat(74));
    let pid = platform::current::get_window_pid(handle);
    println!(
        "  Window:  {handle}  process {}",
        if platform::current::get_process_name(pid).is_empty() {
            "?".to_string()
        } else {
            platform::current::get_process_name(pid)
        }
    );
    println!("  Observing for {seconds:.0}s at {}px...", cfg.frame_size);

    let mut grabber = platform::FrameGrabber::new(handle);
    let mut stack = FrameStack::new(cfg.frame_size, cfg.frame_stack);
    let mut deltas: Vec<f32> = Vec::new();
    let mut failures = 0u32;
    let mut frames = 0u32;
    let mut black = 0u32;
    let mut previous: Option<Vec<f32>> = None;
    let pace = std::time::Duration::from_secs_f32(1.0 / cfg.target_fps.max(1.0));
    let started = std::time::Instant::now();

    while started.elapsed().as_secs_f32() < seconds {
        let Some(raw) = grabber.grab() else {
            failures += 1;
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        };
        let observation = stack.push(&raw);
        let motion = observation.channel(cfg.frame_stack);
        let mean_motion = mean(motion);
        if previous.is_some() {
            deltas.push(mean_motion);
        }
        // `observation[0]` is the oldest frame in the ring, normalized.
        if mean(observation.channel(0)) < 0.01 {
            black += 1;
        }
        previous = Some(observation.data.clone());
        frames += 1;
        std::thread::sleep(pace);
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "  Captured {frames} frame(s) in {elapsed:.1}s ({:.1} fps), {:.1} ms per grab",
        frames as f64 / elapsed.max(1e-6),
        grabber.mean_ms()
    );
    if failures > 0 {
        println!("  {failures} grab(s) returned nothing.");
        if let Some(error) = &grabber.last_error {
            println!("  last error: {error}");
        }
    }

    if frames < 3 {
        println!(
            "  [FAIL] The window is not giving up any frames. Is it minimized, or is the \
             game still loading?"
        );
        return 1;
    }
    let mean_delta = mean(&deltas);
    let changing = 100.0
        * deltas.iter().filter(|delta| **delta > 1e-3).count() as f64
        / deltas.len().max(1) as f64;
    if black as f64 > frames as f64 * 0.9 {
        println!(
            "  [FAIL] Almost every frame is black. A hardware-accelerated game sometimes \
             refuses PrintWindow; try borderless windowed mode, or run the game windowed."
        );
        return 1;
    }
    if mean_delta < 1e-4 {
        println!(
            "  [WARN] The image is not changing at all. The bot will learn nothing from a \
             still screen: the game is probably paused while unfocused, or the window is \
             showing a loading screen."
        );
    } else {
        println!(
            "  [OK]   Image is live: mean change {mean_delta:.4}, changing on {changing:.0}% \
             of frames."
        );
    }
    println!("{}", "-".repeat(74));
    println!();
    0
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Every window the bot can see, and which of them it will not drive.
pub fn list_windows() -> i32 {
    for line in platform::capability_report() {
        println!("[Platform] {line}");
    }
    let windows = platform::current::enumerate_windows(platform::MIN_AREA);
    if windows.is_empty() {
        println!("No windows found.");
        return 1;
    }
    println!();
    println!("{}", "-".repeat(78));
    println!("  WINDOWS (pass one to --window HANDLE)");
    println!("{}", "-".repeat(78));
    println!("{}", platform::format_window_table(&windows));
    println!("{}", "-".repeat(78));
    let games: Vec<&WindowInfo> = windows
        .iter()
        .filter(|window| platform::looks_like_game(window))
        .collect();
    match games.first() {
        Some(game) => {
            println!(
                "  Largest likely game: {} ({}) hwnd={}",
                game.title, game.process, game.handle
            );
            println!("  That is what the bot picks automatically.");
        }
        None => println!("  Nothing here looks like a game; the bot will ask you to pick."),
    }
    println!("  Marked windows are never auto-selected: the terminal this bot runs");
    println!("  in, its own console, and programs that are not games.");
    println!();
    0
}

/// Straighten out a key left down, and every mouse button with it.
pub fn release_all_keys(cfg: &Config, handle: Handle) -> i32 {
    let keymap = Keymap::load(&cfg.keymap_path).unwrap_or_else(Keymap::default_layout);
    let mut injector = platform::Injector::new(handle, cfg.focus_policy);
    println!("[Release] Lifting every key in the whitelist and all mouse buttons.");
    injector.acquire_focus(true);
    injector.begin_action();
    for code in keymap.vks.values() {
        injector.release_vk(*code);
    }
    for code in [0x01u32, 0x02, 0x04] {
        injector.release_vk(code);
    }
    injector.end_action();
    println!("[Release] Done.");
    0
}

/// What the bot may do, in full: the whitelist and the action space it implies.
pub fn show_actions(cfg: &Config, mut keymap: Keymap) -> i32 {
    keymap.set_mouse_turn(if cfg.mouse_turn_pixels == 0 {
        None
    } else {
        Some(cfg.mouse_turn_pixels)
    });
    let space = ActionSpace::build(
        &keymap,
        cfg.max_held_keys,
        &cfg.speed_levels,
        true,
        true,
    );
    let base = keymap.default_mouse_turn.max(1);

    println!();
    println!("{}", "-".repeat(74));
    println!("  ACTION SPACE");
    println!("{}", "-".repeat(74));
    println!(
        "  Keymap: {}",
        keymap.path.clone().unwrap_or_else(|| "(built-in defaults)".to_string())
    );
    println!("  {}", keymap.describe());
    println!("  {}", space.describe());
    println!();
    println!(
        "  Held keys, each decided independently (at most {} at once):",
        space.max_held
    );
    for (index, name) in space.hold_names.iter().enumerate() {
        println!("    [{index}] {name}");
    }
    println!();
    println!("  Turning (two choices per step: which way, and how fast).");
    println!("  The base step is the bot's own: it starts at {base}px and the");
    println!(
        "  bot moves it inside [{}, {}] pixels as it learns this game.",
        cfg.mouse_turn_min, cfg.mouse_turn_max
    );
    println!("    directions:");
    for (index, (name, delta)) in space.directions.iter().enumerate() {
        let rendered = if *delta == (0, 0) {
            "do not turn".to_string()
        } else {
            format!("move mouse by {:+}, {:+} per 1x", delta.0, delta.1)
        };
        println!("      [{index}] {name:<9} {rendered}");
    }
    println!("    speeds (multipliers on the base step):");
    for (index, level) in space.speed_levels.iter().enumerate() {
        let pixels = ((base as f32) * level).round().max(1.0) as i32;
        println!("      [{index}] x{level:<5} -> {pixels:+}px from the current {base}px base");
    }
    println!();
    println!("  Tap / click (one choice per step):");
    for (index, tap) in space.taps.iter().enumerate() {
        println!("    [{index}] {}", tap.label);
    }
    println!("{}", "-".repeat(74));
    println!();
    0
}

/// Measure what one decision actually costs, on this machine, for this window.
///
/// The Python's benchmark, and the only honest way to answer "is it fast
/// enough?": the model and the capture are timed separately because they have
/// completely different fixes - a smaller window, or a smaller model.
pub fn benchmark<B: burn::tensor::backend::AutodiffBackend>(
    cfg: &Config,
    keymap: &Keymap,
    handle: Handle,
    decisions: usize,
    device: B::Device,
) -> i32 {
    let space = ActionSpace::build(keymap, cfg.max_held_keys, &cfg.speed_levels, true, true);
    // A dry run: this measures the loop, and injecting nothing is not only safer
    // but the only way to time the model rather than the game's reaction to it.
    //
    // Pacing is switched off for the measurement. The Python times `session.step`
    // with the pacer running, and the pacer's job is to *sleep* until the next
    // slot - so its "env + capture" number is mostly the sleep it asked for, and
    // its comparison against a per-step budget can essentially never pass. What a
    // user wants to know is what the work costs, and a deliberate wait is not
    // work.
    let mut bench_cfg = cfg.clone();
    bench_cfg.target_fps = 0.0;
    bench_cfg.capture_delay = 0.0;
    let env = match crate::game::RealGameEnv::new(&bench_cfg, handle, &space, true) {
        Ok(env) => env,
        Err(error) => {
            println!("[Bench] Could not open the window: {error}");
            return 1;
        }
    };
    let mut session = match crate::session::GameSession::<B>::new(
        &bench_cfg,
        Box::new(env),
        space,
        Some(keymap),
        device,
    ) {
        Ok(session) => session,
        Err(error) => {
            println!("[Bench] Could not build a session: {error}");
            return 1;
        }
    };

    println!();
    println!("{}", "-".repeat(74));
    println!("  BENCHMARK (dry run: nothing is injected)");
    println!("{}", "-".repeat(74));
    println!("  {}", cfg.describe());

    // Warm up: the first forward pass initialises buffers that later ones reuse.
    for _ in 0..15 {
        let action = session.decide();
        session.step(&action);
    }
    let mut model_times: Vec<f64> = Vec::with_capacity(decisions);
    let mut env_times: Vec<f64> = Vec::with_capacity(decisions);
    let mut action = session.decide();
    for _ in 0..decisions {
        let started = std::time::Instant::now();
        action = session.decide();
        model_times.push(started.elapsed().as_secs_f64() * 1000.0);
        let started = std::time::Instant::now();
        session.step(&action);
        env_times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(|a, b| a.total_cmp(b));
        values.get(values.len() / 2).copied().unwrap_or(0.0)
    };
    let model_ms = median(&mut model_times);
    let env_ms = median(&mut env_times);
    let total_ms = model_ms + env_ms;
    println!("  decision + model : {model_ms:6.2} ms");
    println!(
        "  env + capture    : {env_ms:6.2} ms  ({} game step(s), no pacing)",
        cfg.action_repeat
    );
    println!(
        "  total per step   : {total_ms:6.2} ms  ({:.1} decisions/s)",
        1000.0 / total_ms.max(1e-6)
    );
    // One decision covers `action_repeat` game steps, so the budget it has to
    // fit is that many step budgets - not one. Comparing a per-decision cost
    // against a per-step budget is what makes the Python's version report
    // "too slow" on a machine that is keeping up.
    let step_budget_ms = 1000.0 / cfg.target_fps.max(1.0) as f64;
    let decision_budget_ms = step_budget_ms * cfg.action_repeat.max(1) as f64;
    println!(
        "  target           : {step_budget_ms:6.2} ms per game step  ({decision_budget_ms:.2} ms \
         per decision at action_repeat {})",
        cfg.action_repeat.max(1)
    );
    if total_ms > decision_budget_ms {
        println!(
            "  [WARN] Too slow for the target rate ({total_ms:.0} ms of \
             {decision_budget_ms:.0} ms). The bot will still run; it will react late."
        );
        if env_ms > model_ms {
            println!(
                "         The time is in capture ({env_ms:.0} ms), not the model ({model_ms:.0} \
                 ms). A big window costs more to grab than a small one; run the game windowed, \
                 or lower target_fps."
            );
            println!("         Cross-check with: bot1 --check-capture");
        } else {
            println!(
                "         The time is in the model ({model_ms:.0} ms). Lower embed_dim (now \
                 {}), mem_tokens (now {}), transformer_layers (now {}) or frame_size (now {}).",
                cfg.embed_dim, cfg.mem_tokens, cfg.transformer_layers, cfg.frame_size
            );
        }
    } else {
        println!("  [OK]   Fast enough for the target rate.");
    }
    println!("{}", "-".repeat(74));
    println!();
    let _ = action;
    session.close();
    0
}

/// The advice worth giving about an action space, given what it contains.
pub fn action_space_advice(cfg: &Config, keymap: &Keymap, space: &ActionSpace) -> Vec<String> {
    let mut advice = Vec::new();
    let holds: Vec<String> = space
        .hold_names
        .iter()
        .map(|name| name.to_uppercase())
        .collect();
    let movement = ["W", "A", "S", "D"];
    if !holds.iter().any(|hold| movement.contains(&hold.as_str())) {
        let listed = if holds.is_empty() {
            "-".to_string()
        } else {
            holds.join(", ")
        };
        advice.push(format!(
            "No movement key in the whitelist: hold=[{listed}]. A bot with no way to walk \
             cannot make progress on any game - it will press what it has and look like it \
             is shaking on the spot."
        ));
        advice.push(format!(
            "Fix: re-run --calibrate and spend the recording actually walking around, or \
             edit '{}' to add W/A/S/D (or whatever this game calls forward).",
            cfg.keymap_path
        ));
    }
    if keymap.default_mouse_turn < 4 {
        advice.push(format!(
            "The measured mouse turn step is {} px, which is the signature of a game that \
             locks the cursor: the recorder cannot see the real look speed, so the \
             measurement is a twitch. The bot now sets its own speed - it starts from this \
             number, scales it up when a turn does not move the view, and chooses the \
             multiplier per decision - so this is not fatal. If the view still never turns, \
             pass --mouse-turn 20 (or more) to start it higher.",
            keymap.default_mouse_turn
        ));
    }
    if space.taps.len() > 6 {
        advice.push(format!(
            "{} tap/click choices are enabled, so a large share of every rollout is spent on \
             clicking things. In a game with an inventory or an attack button this is mostly \
             wasted input.",
            space.taps.len()
        ));
    }
    advice
}

/// Everything worth telling the user before a run starts.
pub fn preflight_warnings(
    cfg: &Config,
    keymap: Option<&Keymap>,
    target: Option<&WindowInfo>,
) -> Vec<String> {
    let mut warnings = Vec::new();
    if cfg.torch_threads > 2 {
        warnings.push(format!(
            "The tensor backend is using {} threads; on a small model this is usually slower \
             than 1. Set BOT_TORCH_THREADS=1 to compare.",
            cfg.torch_threads
        ));
    }
    match keymap {
        None => warnings.push(
            "No keymap file: using built-in defaults. Run --calibrate once so the bot may \
             only press the keys you actually use."
                .to_string(),
        ),
        Some(keymap) if keymap.holds.is_empty() => warnings.push(
            "The keymap has no hold keys, so the bot cannot walk or hold anything down."
                .to_string(),
        ),
        _ => {}
    }
    if let Some(target) = target {
        if platform::current::is_non_game_process(&target.process.to_lowercase()) {
            warnings.push(format!(
                "The selected window is {}, which is not a game. Every key the bot presses \
                 will go to that program instead of the game. Check --list-windows, then \
                 restart with --window HANDLE.",
                platform::describe_window(Some(target))
            ));
        }
        if let Some(keymap) = keymap
            && let Some(recorded) = keymap.game_hwnd
            && recorded as Handle != target.handle
        {
            warnings.push(format!(
                "--calibrate recorded the game as '{}' (handle {recorded}), but this run \
                 selected {}. If the recorded handle is still the game, restart with \
                 --window {recorded}.",
                keymap.game,
                platform::describe_window(Some(target))
            ));
        }
    }
    if let Some(keymap) = keymap
        && keymap.default_mouse_turn < 4
    {
        warnings.push(format!(
            "The mouse turn step is {} pixel(s). A game that locks the cursor hides your real \
             look speed from the recorder. The bot starts from this number and scales it up \
             on its own when turns do not move the view, but if you already know the right \
             step, --mouse-turn sets it.",
            keymap.default_mouse_turn
        ));
    }
    warnings
}

/// [`preflight_warnings`], plus the advice the action space itself provokes.
pub fn preflight(
    cfg: &Config,
    keymap: Option<&Keymap>,
    target: Option<&WindowInfo>,
) -> Vec<String> {
    let mut warnings = preflight_warnings(cfg, keymap, target);
    if let Some(keymap) = keymap {
        let space = ActionSpace::build(
            keymap,
            cfg.max_held_keys,
            &cfg.speed_levels,
            true,
            true,
        );
        warnings.extend(action_space_advice(cfg, keymap, &space));
    }
    warnings
}
