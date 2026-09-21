//! The acceptance test for the port's learning path.
//!
//! `--selftest` is the claim that the identical algorithm learns a game it has
//! never seen. Running it here means a change to the model, the reward, the
//! rollout or the PPO update that quietly breaks learning fails the suite
//! instead of failing a user's run hours later.
//!
//! It is `#[ignore]`d because it is a real training run (about half a minute in
//! release, several minutes in a debug build), and `cargo test` should stay
//! quick. Run it with:
//!
//! ```text
//! cargo test --release --test selftest -- --ignored --nocapture
//! ```

use burn::backend::{Autodiff, Flex};

type Backend = Autodiff<Flex>;

#[test]
#[ignore = "trains for a few thousand steps; run with --release -- --ignored"]
fn the_corridor_self_test_still_learns() {
    let device = Default::default();
    let result = bot1::synth::run_learning_check::<Backend>(
        device,
        true,
        0,
        24_000,
        48,
        1024,
        60,
    )
    .expect("the self-test itself must run");

    // The corridor check is the one that means "it learned to play".
    let corridor = result.corridor.as_ref().expect("a corridor check");
    assert!(
        corridor.passed,
        "the corridor check failed: distance {:.1} -> {:.1}, random baseline {:.1}",
        corridor.before, corridor.after, corridor.random_baseline
    );
    assert!(
        corridor.after > 1.5 * corridor.random_baseline,
        "the policy must beat a random walk by a clear margin: {:.1} against {:.1}",
        corridor.after,
        corridor.random_baseline
    );
    // And the reward must still prefer progress to dying or standing still.
    assert!(
        result.control_passed,
        "walking backwards earned {:.0}% of walking forwards",
        result.backwards_ratio * 100.0
    );
    assert!(
        result.idle_passed,
        "standing still earned {:.0}% of walking forwards",
        result.idle_ratio * 100.0
    );
    assert!(result.ok, "the self-test as a whole failed");
}
