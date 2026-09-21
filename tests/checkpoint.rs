//! A checkpoint must give back exactly the run that was saved.
//!
//! This is the test that makes `--resume` trustworthy. It does not check that a
//! file was written; it drives a real session on the synthetic corridor, saves,
//! builds a *second* session from scratch, loads, and compares the two policies'
//! own output on the same observation. A transposed weight, a misnamed tensor or
//! a stale counter all fail here, and none of them would fail a "did it save?"
//! test.

use bot1::model::ActorCritic;
use bot1::session::GameSession;
use bot1::synth::{SyntheticEnv, synthetic_action_space, synthetic_train_config};
use bot1::vision::Observation;
use burn::backend::{Autodiff, Flex};
use burn::prelude::*;
use burn::tensor::TensorData;

type Backend = Autodiff<Flex>;
type Inner = Flex;

/// The policy's own opinion about one observation: hold-key logits and value.
///
/// Taken through the collection copy of the weights, which is the model that
/// actually plays, so this compares what the bot would *do* rather than merely
/// what it stores.
fn opinion(session: &GameSession<Backend>, observation: &Observation) -> Vec<f32> {
    let device = Default::default();
    let input = Tensor::<Inner, 4>::from_data(
        TensorData::new(
            observation.data.clone(),
            [
                1,
                observation.channels,
                observation.height,
                observation.width,
            ],
        ),
        &device,
    );
    let hidden = session.acting.initial_hidden(1, &device);
    let tokens = session.acting.encode(input);
    let (_window, context, features) = session.acting.forward(hidden, tokens);
    let (hold, turn, _speed, _tap, value) = session.acting.heads(features, context);
    let mut out = hold.into_data().to_vec::<f32>().unwrap();
    out.extend(turn.into_data().to_vec::<f32>().unwrap());
    out.extend(value.into_data().to_vec::<f32>().unwrap());
    out
}

fn build(cfg: &bot1::Config) -> GameSession<Backend> {
    let device = Default::default();
    let env = SyntheticEnv::new(cfg, 20, 100);
    GameSession::<Backend>::new(
        cfg,
        Box::new(env),
        synthetic_action_space(cfg),
        None,
        device,
    )
    .expect("a session on the synthetic corridor")
}

#[test]
fn a_session_round_trips_through_a_checkpoint() {
    let cfg = synthetic_train_config(48, 64, 3);
    let mut session = build(&cfg);
    session.reset(Some(7), false);
    // Enough steps that the counters, the reward statistics and the observation
    // stack all have something to lose.
    for _ in 0..60 {
        let action = session.decide();
        session.step(&action);
    }
    let probe = session.observation.clone();
    let before = opinion(&session, &probe);
    let saved_step = session.total_steps;
    let saved_episodes = session.episodes;
    let saved_mouse = session.policy.mouse_step;
    assert!(saved_step >= 60);

    let directory = std::env::temp_dir().join("bot1-checkpoint-round-trip");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("checkpoint.safetensors");
    session.save_checkpoint(&path).expect("the checkpoint saves");
    assert!(path.exists());

    // A session built from scratch knows nothing; loading must give it
    // everything.
    let mut restored = build(&cfg);
    assert_eq!(restored.total_steps, 0);
    let meta = restored
        .load_checkpoint(&path)
        .expect("the checkpoint loads");

    assert_eq!(meta.step, saved_step);
    assert_eq!(restored.total_steps, saved_step, "the step counter resumes");
    assert_eq!(restored.episodes, saved_episodes, "the episode count resumes");
    assert!(
        (restored.policy.mouse_step - saved_mouse).abs() < 1e-6,
        "the bot's own mouse step is part of what it learned"
    );
    assert!(
        (restored.reward.error_running.mean - session.reward.error_running.mean).abs() < 1e-9,
        "the running estimate the curiosity term is measured against resumes too"
    );
    assert!(
        (restored.reward.detector.ambient - session.reward.detector.ambient).abs() < 1e-9,
        "the ambient motion estimate resumes"
    );

    let after = opinion(&restored, &probe);
    assert_eq!(
        before.len(),
        after.len(),
        "the same shapes come back"
    );
    for (index, (left, right)) in before.iter().zip(&after).enumerate() {
        assert_eq!(
            left, right,
            "output {index} differs after a save and load: {left} against {right}"
        );
    }
}

#[test]
fn a_checkpoint_from_a_different_action_space_is_refused() {
    let cfg = synthetic_train_config(48, 64, 3);
    let session = build(&cfg);
    let directory = std::env::temp_dir().join("bot1-checkpoint-mismatch");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("checkpoint.safetensors");
    session.save_checkpoint(&path).expect("the checkpoint saves");

    // A session whose action space is *wider*: the real nine-key default layout
    // rather than the synthetic corridor's three. Its heads do not fit, and a
    // loader that assigned them anyway would produce a model that runs and
    // plays nonsense.
    let device = Default::default();
    let keymap = bot1::keys::Keymap::default_layout();
    let wide = bot1::keys::ActionSpace::build(
        &keymap,
        cfg.max_held_keys,
        &cfg.speed_levels,
        true,
        true,
    );
    assert_ne!(wide.head_sizes(), session.action_space.head_sizes());
    let mut mismatched = GameSession::<Backend>::new(
        &cfg,
        Box::new(SyntheticEnv::new(&cfg, 20, 100)),
        wide,
        None,
        device,
    )
    .expect("a session with a wider action space");
    let error = mismatched
        .load_checkpoint(&path)
        .expect_err("a different action space must be refused");
    assert!(
        error.to_string().contains("action space"),
        "the reason should name the mismatch, got: {error}"
    );
}

/// `to_weights` and `load` must be exact inverses, checked on the real
/// architecture rather than on the synthetic one.
#[test]
fn the_exported_weights_load_back_identical() {
    let cfg = bot1::Config::default();
    let device = Default::default();
    let mut model =
        ActorCritic::<Inner>::new(&cfg, [10, 5, 5, 4], cfg.channels(), &device);
    let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(11);
    model.init(&mut rng, &device);

    let observation = Observation::zeros(cfg.channels(), cfg.frame_size, cfg.frame_size);
    let input = Tensor::<Inner, 4>::from_data(
        TensorData::new(
            (0..observation.data.len())
                .map(|index| (index % 251) as f32 / 251.0)
                .collect::<Vec<f32>>(),
            [1, cfg.channels(), cfg.frame_size, cfg.frame_size],
        ),
        &device,
    );
    let before = model
        .encode(input.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let directory = std::env::temp_dir().join("bot1-weights-round-trip");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("weights.safetensors");
    bot1::weights::write_safetensors(&path, &model.to_weights()).unwrap();

    let mut restored =
        ActorCritic::<Inner>::new(&cfg, [10, 5, 5, 4], cfg.channels(), &device);
    restored
        .load(&bot1::weights::Weights::open(&path).unwrap(), &device)
        .expect("the exported weights load");
    let after = restored
        .encode(input)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    assert_eq!(before.len(), after.len());
    for (index, (left, right)) in before.iter().zip(&after).enumerate() {
        assert_eq!(
            left, right,
            "value {index} differs after an export and re-import: {left} against {right}"
        );
    }
}
