# bot1 — a game-agnostic bot that learns to play from pixels

A small reinforcement-learning bot that watches a game window, presses keys
through a calibrated whitelist, and learns — with no knowledge of the game, no
reading of a score, and no GPU.

```bash
python bot1.py --selftest    # prove it learns, on a game it has never seen
python bot1.py --calibrate    # show it which keys you use (once)
python bot1.py                # train on your game
```

## What it does

* **Sees** the game as grey frames plus a motion channel, at whatever
  resolution you configure (96px by default).
* **Acts** through a factored action space: which keys to hold (each decided
  independently, capped at four at once), how far to turn the mouse, and one
  optional tap or click. Around 1700 combinations from a 9-key whitelist,
  without an enumeration that can never be explored.
* **Learns** with PPO on a recurrent (CNN + GRU) policy, about 400k parameters.
* **Rewards itself** intrinsically — see below.
* **Checkpoints** every few minutes and resumes automatically. `F8` pause,
  `F9` save, `F10` quit, `Ctrl+C` saves and quits.

## Why it does not stall

The previous design degraded after a while, for concrete reasons. Each one is
answered here, and each is measured rather than assumed:

| Problem | What this does |
| --- | --- |
| A transformer over a stack of frames on every control step | A CNN encodes each frame once; a GRU carries memory. ~3 ms per decision on a 12-thread laptop CPU. |
| A rollout of **one** transition per update, so normalised advantage was exactly zero and the gradient was silently nothing | 1024-step rollouts, 16-step recurrent minibatches, and a diagnostic that reports the critic's explained variance. |
| Unbounded novelty tables and an O(n) trim that eventually blocked for seconds | Every memory has a fixed capacity with O(1) eviction, and the per-step and per-update cost is reported. |
| Intrinsic terms with a broken running-scale estimate, making novelty ~10× louder than intended | Exponential running scales with a floor, so a novelty bonus starts near 1.0 and stays comparable for the life of the run. |
| A policy that drifts onto "press nothing" and looks fine in the log | Per-step histogram, an idle charge that ramps, and an explicit diagnosis when the policy stops pressing anything. |
| A game that stops rendering while unfocused, so it trains on a still image forever | A frozen-screen watchdog that says exactly that, once. |

Run `python bot1.py --benchmark` and it will tell you the per-decision cost on
your machine and whether it fits your target rate.

## The reward

Entirely intrinsic; nothing reads the game's score, and there is no manual
feedback. Four terms:

* **Episodic novelty** — a state not reached since the last reset pays, and
  pays less each time it is revisited. This is the term that means *progress*.
* **Depth** — a floor paid every step, scaled by how much new ground the
  current episode has covered. This is what makes pressing on better than
  standing still rather than merely different.
* **RND novelty** — prediction error of a fixed random network, kept quiet
  because raw prediction error is largest where the screen is most chaotic.
* **Charges** — for pressing nothing, and for repeating one screen.

Resets are **inferred**, because a bot that knows nothing about the game cannot
be told it died. Three independent tests: a large change the forward model
cannot account for, a large change no action caused, and a return to the
state the episode started in. This matters more than it sounds — if a death is
not recognised as a reset, the novelty memory is never cleared and *dying
becomes a source of new states*, so the bot learns to die.

## Does it actually learn?

`--selftest` is the answer, and it is a real test rather than a demo. The
identical model, reward and PPO update are trained on a synthetic game whose
progress is known — a long corridor where only moving forward makes progress —
and the report includes a random-action baseline, because in a small world a
random walk scores well by accident.

```
  corridor (go forward)    distance   0.0 ->  98.0   random   7.2
  always forward           reward/step +0.6480
  always backwards         reward/step +0.1970    30% of forward
  never press anything     reward/step +0.1190    18% of forward
```

The last two lines are controls: if going backwards or standing still earned as
much as going forwards, the reward would not be a progress signal no matter how
well the first line looked.

## What "learns to play" means here, honestly

With a purely intrinsic reward and no access to the game's score, the bot
optimises for **reaching states it has not reached before**, which is the
generic form of making progress. It will explore a game thoroughly, learn which
buttons do what, and get further and further in. It will not discover "win the
level" as a concept, because nothing tells it that winning exists.

If you want it optimising a real objective, the honest route is to give it one:
a score region, a manual reward key, or a game-specific signal. The reward
module is a single function (`botcore/novelty.py`, `IntrinsicReward.compute`)
with a documented component breakdown, so that is a small change rather than a
rewrite.

## Layout

```
bot1.py              entry point; --help lists every flag
botcore/
  config.py          every tunable, with the reasoning next to it
  capture.py         window finding, frame grabbing, the observation stack
  keys.py            the calibrated whitelist, action space, input injection
  model.py           CNN encoder, GRU policy with factored heads, RND
  novelty.py         the intrinsic reward and the reset detector
  replay.py          fixed-size rollout storage and recurrent minibatching
  trainer.py         PPO, with a measured update budget
  session.py         the loop, the health checks, checkpointing
  game.py            the real game window as an environment
  synth.py           the synthetic game used by --selftest
  calibrate.py       the key recorder behind --calibrate
  diagnostics.py     --list-windows, --check-capture, --benchmark, --release
  runtime.py         checkpoints, hotkeys, Ctrl+C, process priority
  cli.py             argument parsing and the mode dispatch
```

## Requirements

Python 3.10+, Windows (capture and input use Win32), and:

```bash
python -m pip install -r requirements.txt
```

CPU only, by design. There is no CUDA path because there is nothing here big
enough to need one.
