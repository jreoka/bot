//! bot1, in Rust.
//!
//! Stage 2 of the port: the model, the reward, PPO and the synthetic game are
//! all in place, and `--selftest` runs the real algorithm against a game whose
//! progress is known. The Win32 layer and the session's checkpointing arrive
//! next, so today this binary runs the self-test and says what it is waiting for
//! otherwise.

use burn::backend::{Autodiff, Flex};

type Backend = Autodiff<Flex>;

fn main() -> anyhow::Result<()> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.iter().any(|argument| argument == "--help" || argument == "-h") {
        print_help();
        return Ok(());
    }
    let value_of = |name: &str, fallback: u64| -> u64 {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(fallback)
    };
    if arguments.iter().any(|argument| argument == "--selftest") {
        let device = Default::default();
        let result = bot1::synth::run_learning_check::<Backend>(
            device,
            true,
            value_of("--seed", 0),
            value_of("--steps", 24_000),
            48,
            1024,
            60,
        )?;
        std::process::exit(if result.ok { 0 } else { 1 });
    }
    if arguments.iter().any(|argument| argument == "--list-windows") {
        return code(bot1::diagnostics::list_windows());
    }
    if arguments.iter().any(|argument| argument == "--platform") {
        return platform_report();
    }
    if arguments.iter().any(|argument| argument == "--actions") {
        let cfg = bot1::Config::default();
        let keymap = bot1::keys::Keymap::load(&cfg.keymap_path)
            .unwrap_or_else(bot1::keys::Keymap::default_layout);
        return code(bot1::diagnostics::show_actions(&cfg, keymap));
    }
    if arguments.iter().any(|argument| argument == "--check-capture") {
        return check_capture(&arguments, value_of("--window", 0));
    }
    if arguments.iter().any(|argument| argument == "--hotkeys") {
        return hotkeys(value_of("--seconds", 3));
    }
    if arguments.iter().any(|argument| argument == "--benchmark") {
        return benchmark(value_of("--window", 0), value_of("--decisions", 120));
    }
    if arguments.iter().any(|argument| argument == "--calibrate") {
        return calibrate(value_of("--window", 0), value_of("--seconds", 0));
    }
    // Anything that is not one of the modes above is a run, which is what the
    // Python does with no flags at all.
    return train(&arguments);
}

/// The default mode: drive a window, learn from it, checkpoint, and stop
/// cleanly on Ctrl+C.
fn train(arguments: &[String]) -> anyhow::Result<()> {
    use bot1::control::Control as _;
    use bot1::platform;
    let has = |name: &str| arguments.iter().any(|argument| argument == name);
    let value_of = |name: &str, fallback: u64| -> u64 {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(fallback)
    };

    let dry_run = has("--dry-run");
    let mut cfg = bot1::Config {
        seed: value_of("--seed", 0),
        enable_hotkeys: !has("--no-hotkeys"),
        wait_for_start: !has("--start-now"),
        // On by default: a game that frees the cursor (an inventory, a pause
        // menu) lets the bot's own mouse steps walk the pointer out of the
        // window, and the next click lands on whatever is under it.
        clip_cursor: !has("--no-clip-cursor"),
        ..bot1::Config::default()
    }
    .env_overrides();
    if let Some(index) = arguments.iter().position(|a| a == "--focus") {
        cfg.focus_policy = match arguments.get(index + 1).map(String::as_str) {
            Some("always") => bot1::config::FocusPolicy::Always,
            Some("never") => bot1::config::FocusPolicy::Never,
            _ => bot1::config::FocusPolicy::Once,
        };
    }
    if let Some(index) = arguments.iter().position(|a| a == "--fps") {
        if let Some(fps) = arguments.get(index + 1).and_then(|v| v.parse::<f32>().ok()) {
            cfg.target_fps = fps;
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--checkpoint-every") {
        if let Some(seconds) = arguments
            .get(index + 1)
            .and_then(|value| value.parse::<f32>().ok())
        {
            cfg.checkpoint_interval_sec = seconds;
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--checkpoint-dir") {
        if let Some(directory) = arguments.get(index + 1) {
            cfg.checkpoint_dir = directory.clone();
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--keep") {
        if let Some(keep) = arguments
            .get(index + 1)
            .and_then(|value| value.parse::<usize>().ok())
        {
            cfg.checkpoint_keep = keep;
        }
    }
    // The tuning flags: the same tunables the config carries, and the same ones
    // the `BOT_*` environment overrides reach. A macro rather than twenty copies
    // of the same six lines, so a flag cannot end up wired to the wrong field.
    macro_rules! numeric_flags {
        ($($flag:literal => $field:ident : $ty:ty),* $(,)?) => {
            $(
                if let Some(index) = arguments.iter().position(|a| a == $flag) {
                    match arguments.get(index + 1).and_then(|value| value.parse::<$ty>().ok()) {
                        Some(value) => cfg.$field = value,
                        None => println!(
                            "[Config] {} needs a {}; ignoring it.",
                            $flag,
                            stringify!($ty)
                        ),
                    }
                }
            )*
        };
    }
    numeric_flags! {
        "--frame-size" => frame_size: usize,
        "--frame-stack" => frame_stack: usize,
        "--action-repeat" => action_repeat: usize,
        "--max-held-keys" => max_held_keys: usize,
        "--embed-dim" => embed_dim: usize,
        "--layers" => transformer_layers: usize,
        "--memory-window" => mem_tokens: usize,
        "--patch-size" => patch_size: usize,
        "--patch-width" => patch_width: usize,
        "--heads" => attention_heads: usize,
        "--ffn-hidden" => ffn_hidden: usize,
        "--seq-len" => seq_len: usize,
        "--rollout" => rollout_steps: usize,
        "--minibatch" => minibatch_size: usize,
        "--epochs" => epochs_per_update: usize,
        "--lr" => adam_lr: f32,
        "--entropy" => entropy_coef: f32,
        "--clip" => clip_eps: f32,
        "--gamma" => gamma: f32,
        "--update-budget" => update_seconds_budget: f32,
        "--status-every" => status_every: usize,
        "--mouse-turn" => mouse_turn_pixels: i32,
        "--mouse-adapt-rate" => mouse_adapt_rate: f32,
        "--tap-seconds" => tap_seconds: f32,
        "--torch-threads" => torch_threads: usize,
    }
    if let Some(index) = arguments.iter().position(|a| a == "--mouse-speed-levels") {
        match arguments.get(index + 1) {
            Some(raw) => {
                let levels: Vec<f32> = raw
                    .replace(',', " ")
                    .split_whitespace()
                    .filter_map(|part| part.parse::<f32>().ok())
                    .collect();
                if levels.is_empty() {
                    println!(
                        "[Config] --mouse-speed-levels '{raw}' is not a list of numbers; using \
                         the default."
                    );
                } else {
                    cfg.speed_levels = levels;
                }
            }
            None => println!("[Config] --mouse-speed-levels needs a list, e.g. 0.5,1,2."),
        }
    }
    if let Some(index) = arguments.iter().position(|a| a == "--log") {
        cfg.log_json = arguments.get(index + 1).cloned();
    }
    if let Some(index) = arguments.iter().position(|a| a == "--priority") {
        cfg.priority = match arguments.get(index + 1).map(String::as_str) {
            Some("normal") => bot1::config::Priority::Normal,
            Some("high") => bot1::config::Priority::High,
            _ => bot1::config::Priority::BelowNormal,
        };
    }
    cfg.validate()?;

    let keymap = bot1::keys::Keymap::load(&cfg.keymap_path);
    match &keymap {
        Some(keymap) => println!("[Keymap] {}", keymap.describe()),
        None => println!(
            "[Keymap] No '{}': using built-in defaults. Run --calibrate once so the bot may \
             only press the keys you actually use.",
            cfg.keymap_path
        ),
    }
    let keymap = keymap.unwrap_or_else(bot1::keys::Keymap::default_layout);

    let request = platform::TargetRequest {
        configured: if value_of("--window", 0) == 0 {
            None
        } else {
            Some(value_of("--window", 0) as platform::Handle)
        },
        prefer_game_window: !has("--pick"),
        allow_prompt: true,
        remembered_handle: keymap.game_hwnd.map(|handle| handle as platform::Handle),
        remembered_title: keymap.game.clone(),
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        println!("[Run] No window chosen.");
        return code(1);
    };
    let risk = platform::window_risk(Some(&window));
    if !risk.is_empty() && !dry_run {
        println!(
            "[Window] Refusing to drive {}: {risk}.",
            platform::describe_window(Some(&window))
        );
        return code(1);
    }

    let action_space = bot1::keys::ActionSpace::build(
        &keymap,
        cfg.max_held_keys,
        &cfg.speed_levels,
        true,
        true,
    );
    for warning in bot1::diagnostics::preflight(&cfg, Some(&keymap), Some(&window)) {
        println!("[Check] {warning}");
    }
    println!("[Actions] {}", action_space.describe());
    println!("[Run] {}", cfg.describe());
    if dry_run {
        println!("[Run] DRY RUN: nothing will be injected into any window.");
    }

    let env = bot1::game::RealGameEnv::new(&cfg, window.handle, &action_space, dry_run)?;
    let device = Default::default();
    let mut session = bot1::session::GameSession::<Backend>::new(
        &cfg,
        Box::new(env),
        action_space,
        Some(&keymap),
        device,
    )?;

    let mut control = bot1::control::SignalController::new(&cfg);
    control.start();

    // Checkpointing: one manager owns the directory, and the save closure is
    // what the run calls when the timer or the F9 key says it is due.
    let fresh = has("--fresh");
    let mut manager = bot1::checkpoint::CheckpointManager::new(&cfg.checkpoint_dir, cfg.checkpoint_keep);
    if let (true, Some(path)) = (cfg.resume && !fresh, manager.newest()) {
        match session.load_checkpoint(&path) {
            Ok(meta) => println!(
                "[Resume] Continuing {} at step {} ({} episode(s)) from {}.",
                platform::describe_window(Some(&window)),
                meta.step,
                meta.episodes,
                path.display()
            ),
            Err(error) => println!("[Resume] Checkpoint not usable ({error}); starting fresh."),
        }
    } else if !manager.list_kept().is_empty() {
        println!(
            "[Resume] {} checkpoint(s) in '{}' left alone (--fresh).",
            manager.list_kept().len(),
            cfg.checkpoint_dir
        );
    }

    println!(
        "[Run] Playing {}. F8 pauses, F9 checkpoints, F10 quits, Ctrl+C stops and releases \
         every key.",
        platform::describe_window(Some(&window))
    );

    let steps = value_of("--steps", 0);
    let max_decisions = if steps == 0 { None } else { Some(steps) };
    {
        let mut save = |session: &bot1::session::GameSession<Backend>, reason: &str| {
            let meta = session.meta();
            match manager.save(&session.policy.to_weights(), &meta, reason) {
                Ok(path) => println!(
                    "[Checkpoint] step {} ({}): {}",
                    meta.step,
                    reason,
                    path.file_name().unwrap_or_default().to_string_lossy()
                ),
                Err(error) => println!("[Checkpoint] FAILED to save: {error}"),
            }
        };
        session.run(Some(&mut control), max_decisions, Some(&mut save));
        // A run that stopped for any reason still writes what it learned: a run
        // that ends without a checkpoint is a run that lost everything since the
        // last one.
        if session.total_steps > 0 {
            save(&session, "final");
        }
    }
    control.stop();
    session.close();
    println!("[Run] Stopped ({}).", control.stop_reason());
    for entry in manager.list_kept() {
        println!(
            "       step {:>9}  ({})  {}",
            entry.step, entry.reason, entry.path
        );
    }
    Ok(())
}

/// `--benchmark [--window HANDLE] [--decisions N]`: what a step costs here.
fn benchmark(configured: u64, decisions: u64) -> anyhow::Result<()> {
    use bot1::platform;
    let cfg = bot1::Config::default();
    let keymap = bot1::keys::Keymap::load(&cfg.keymap_path)
        .unwrap_or_else(bot1::keys::Keymap::default_layout);
    let request = platform::TargetRequest {
        configured: if configured == 0 {
            None
        } else {
            Some(configured as platform::Handle)
        },
        prefer_game_window: true,
        allow_prompt: true,
        remembered_handle: keymap.game_hwnd.map(|handle| handle as platform::Handle),
        remembered_title: keymap.game.clone(),
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        println!("[Bench] Needs a window: pass --window HANDLE (see --list-windows).");
        return code(1);
    };
    let device = Default::default();
    let status = bot1::diagnostics::benchmark::<Backend>(
        &cfg,
        &keymap,
        window.handle,
        decisions.max(1) as usize,
        device,
    );
    code(status)
}

/// `--hotkeys [--seconds N]`: register the control keys and report what arrived.
///
/// The keys are held by this process for those few seconds, which is exactly
/// what a real run does with them; nothing else is touched.
fn hotkeys(seconds: u64) -> anyhow::Result<()> {
    use bot1::platform;
    let cfg = bot1::Config::default();
    let mut hotkeys = platform::Hotkeys::new(
        &cfg.hotkey_pause,
        &cfg.hotkey_save,
        &cfg.hotkey_quit,
    );
    let registered = hotkeys.start();
    for message in hotkeys.drain_messages() {
        println!("[Hotkeys] {message}");
    }
    if registered {
        println!(
            "[Hotkeys] {} (work from any window)",
            hotkeys.describe()
        );
    } else {
        println!("[Hotkeys] No global hotkeys; Ctrl+C still saves and quits.");
    }
    let started = std::time::Instant::now();
    while started.elapsed().as_secs() < seconds {
        for message in hotkeys.drain_messages() {
            println!("[Hotkeys] {message}");
        }
        for action in hotkeys.poll() {
            println!("[Hotkeys] {} pressed", action.name().to_uppercase());
        }
        std::thread::sleep(std::time::Duration::from_millis(40));
    }
    hotkeys.stop();
    println!("[Hotkeys] Released.");
    Ok(())
}

/// `--calibrate [--window HANDLE] [--seconds N]`: watch the user play, and write
/// down what they actually press.
///
/// Nothing is injected: the hooks only listen, and every keystroke reaches the
/// game unchanged.
fn calibrate(configured: u64, seconds: u64) -> anyhow::Result<()> {
    use bot1::keys::parse_hotkey;
    use bot1::platform;
    let cfg = bot1::Config::default();
    let request = platform::TargetRequest {
        configured: if configured == 0 {
            None
        } else {
            Some(configured as platform::Handle)
        },
        // Not "the largest likely game": the user is about to point at the game
        // themselves, so guessing is worse than asking.
        prefer_game_window: false,
        allow_prompt: true,
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        println!("[Calibrate] No window chosen.");
        return code(1);
    };
    let toggle_vk = parse_hotkey(&cfg.calibrate_key).map(|(_, vk)| vk).unwrap_or(0x77);
    println!();
    println!("{}", "-".repeat(74));
    println!("  KEY CALIBRATION");
    println!("{}", "-".repeat(74));
    println!(
        "  Driving: {}",
        platform::describe_window(Some(&window))
    );
    println!("  Play the game normally. The recorder listens while you do.");
    println!(
        "  Press {} to start recording, {} again to stop and write the keymap.",
        cfg.calibrate_key.to_uppercase(),
        cfg.calibrate_key.to_uppercase()
    );
    println!("  Nothing is sent to the game: the hooks only listen.");
    println!("{}", "-".repeat(74));
    println!();

    let mut recorder = platform::KeyRecorder::start(toggle_vk, window.handle, true);
    if !recorder.installed() {
        println!("[Calibrate] The keyboard hooks could not be installed.");
        return code(1);
    }
    let started = std::time::Instant::now();
    let mut last_report = 0.0f64;
    loop {
        recorder.pump();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = started.elapsed().as_secs_f64();
        if elapsed - last_report >= 5.0 {
            last_report = elapsed;
            println!(
                "  [{elapsed:4.0}s] {} - {} event(s)",
                if recorder.recording() {
                    "RECORDING"
                } else {
                    "waiting  "
                },
                recorder.events()
            );
        }
        // Stop once recording has been switched off at least once.
        if !recorder.recording() && recorder.toggles() >= 1 {
            break;
        }
        if seconds > 0 && elapsed >= seconds as f64 {
            println!("  [Calibrate] Time limit reached.");
            break;
        }
    }
    recorder.stop();

    let recording = recorder.snapshot(Some(&window));
    println!();
    println!("{}", "-".repeat(74));
    println!("  WHAT WAS RECORDED");
    println!("{}", "-".repeat(74));
    println!("{}", recording.report(cfg.calibrate_min_hold as f64));
    println!("{}", "-".repeat(74));
    let keymap = recording.build_keymap(None, cfg.calibrate_min_hold as f64);
    if keymap.holds.is_empty() && keymap.vks.is_empty() {
        println!(
            "[Calibrate] Nothing was recorded, so nothing was saved. Press {} while the \
             recorder is running, then play.",
            cfg.calibrate_key.to_uppercase()
        );
        return code(1);
    }
    let path = keymap.save(&cfg.keymap_path)?;
    println!("[Calibrate] Saved {path}  ({})", keymap.describe());
    bot1::diagnostics::show_actions(&cfg, keymap);
    Ok(())
}

/// `--check-capture [--window HANDLE]`: what the capture actually yields.
fn check_capture(arguments: &[String], configured: u64) -> anyhow::Result<()> {
    use bot1::platform;
    let cfg = bot1::Config::default();
    let request = platform::TargetRequest {
        configured: if configured == 0 {
            None
        } else {
            Some(configured as platform::Handle)
        },
        prefer_game_window: true,
        allow_prompt: true,
        ..Default::default()
    };
    let Some(window) = platform::resolve_target_window(&request) else {
        println!("[Check] No window chosen.");
        return code(1);
    };
    if !platform::is_supported() {
        for line in platform::capability_report() {
            println!("[Platform] {line}");
        }
        return code(1);
    }
    let _ = arguments;
    code(bot1::diagnostics::check_capture(&cfg, window.handle, 5.0))
}

/// Exit with a status, so shell users can chain on it.
fn code(status: i32) -> anyhow::Result<()> {
    std::process::exit(status);
}

/// `--platform`: what this build can and cannot do here, and why.
fn platform_report() -> anyhow::Result<()> {
    use bot1::platform;
    println!();
    println!("  platform: {} ({})", std::env::consts::OS, std::env::consts::ARCH);
    println!(
        "  can drive a window here: {}",
        if platform::is_supported() { "yes" } else { "no" }
    );
    let report = platform::capability_report();
    if report.is_empty() {
        println!("  nothing to report: this platform's window layer is available.");
    }
    for line in report {
        println!("  {line}");
    }
    println!();
    println!(
        "  The model, the reward, the PPO update and --selftest need no window layer and\n\
         \x20 run on every platform this builds for."
    );
    println!();
    Ok(())
}

fn print_help() {
    println!(
        "bot1 {} - a game-agnostic bot that learns to play from pixels\n\
         \n\
         Run it with no flags to train on a game window:\n\
         \n\
         \x20 bot1                     pick a window, wait for the start key, learn\n\
         \x20 bot1 --window HWND       drive a specific window (see --list-windows)\n\
         \x20 bot1 --steps N           stop after N decisions\n\
         \x20 bot1 --dry-run           run the whole loop and inject nothing\n\
         \n\
         Modes:\n\
         \n\
         \x20 --selftest    prove the identical algorithm learns a game it has\n\
         \x20               never seen, on a synthetic corridor whose progress is\n\
         \x20               known, with a random-action baseline and controls\n\
         \x20 --calibrate   watch you play and write down the keys you use\n\
         \x20 --actions     print the whitelist and the action space it implies\n\
         \x20 --list-windows   every window the bot can see, and which it will not drive\n\
         \x20 --check-capture  watch a window for five seconds and report what the\n\
         \x20               capture actually yields\n\
         \x20 --benchmark   what one decision costs here, model against capture\n\
         \x20 --hotkeys     register F8/F9/F10 and report what arrives\n\
         \x20 --platform    what this build can and cannot do on this OS\n\
         \n\
         Options: --seed N, --steps N, --window N, --fps N, --focus once|always|never,\n\
         \x20 --no-hotkeys, --no-clip-cursor, --start-now, --pick, --dry-run, --decisions N.\n\
         \n\
         The cursor is confined to the game window while the bot moves it, so a game\n\
         that frees the cursor (an inventory) cannot have clicks land outside it;\n\
         --no-clip-cursor turns that off, and BOT_CLIP_CURSOR=0 does the same.\n\
         \n\
         Every key is released on exit, including on Ctrl+C. Nothing is sent while the\n\
         game window is not in front, and the bot refuses to drive its own terminal.",
        env!("CARGO_PKG_VERSION")
    );
}
