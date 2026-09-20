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
* **Checkpoints** every few minutes and resumes automatically.

## Starting and stopping

The bot does not touch the game until you tell it to. That is deliberate: a
trainer launched from a terminal that starts injecting straight away types its
first keys into whatever window happens to be in front — the terminal.

```
python bot1.py            # idle; prints the target window and waits
   (click the game so it has focus)
   F8                     # start  (F8 again pauses, F8 again resumes)
   F9                     # checkpoint now
   F10                    # save and quit
   Ctrl+C                 # save and quit
```

Input is sent **only while the game window has focus**. Click the terminal, a
browser, anything else, and the bot immediately lets go of every key and stops
sending until you click back — it never pulls the desktop back to the game
after the initial grab, so focus is yours. `--focus always` restores the old
behaviour of grabbing focus on every action (needed by a few games, and the
reason a terminal could end up fighting for the keyboard); `--focus never`
means the bot never touches focus at all. `--start-now` skips the start key.

Checkpoints are written to `checkpoints/` (an older `checkpoints_v2/` directory
is moved across automatically the first time you run it).

## Which window it plays, and why it will not guess

Input goes to a window, so the first thing a run does is decide which one. It
will not simply take the largest window on the desktop, because that is usually
the terminal it was launched from — and driving your own terminal means typing
the bot's actions into a shell prompt while the game sits untouched, at twenty
decisions a second, with no error printed anywhere. The order is:

1. `--window HWND`, if you passed one.
2. The window the last `--calibrate` recorded: by handle if it is still open
   and still carries the game's title, otherwise by that title — a handle
   Windows has recycled fails the check, and a game gets a new one every
   launch.
3. The window you are looking at, if it looks like a game.
4. The largest window that looks like a game.
5. Otherwise it asks you to pick.

The terminal or IDE the bot runs in, its own console, browsers and editors are
never auto-selected, and the bot refuses to drive them even when you name one
with `--window`. Before it starts it prints the window it is driving, with the
process and handle, and warns when that is not the window you calibrated
against.

## If the game does not move

A working run prints both of these every fifteen seconds, and the pair is the
whole story: one line says the loop is alive, the other says it is playing.

```
[Perf]  20.0 game step(s)/s  = 10.0 decision(s)/s at action_repeat 2 (target 20)
[Input] game window focused - 812 key event(s) and 240 mouse event(s) delivered, 0 step(s) skipped while unfocused
```

So "it says twenty steps a second and nothing happens" is readable straight off
the log:

* `[Input] game window NOT focused` — the game is not the front window. Click
  it; input resumes on its own. The bot never fights you for focus.
* `0 event(s) delivered` with the window focused — the keymap has not pressed
  anything yet (the status block reports a no-op collapse), or the game ignores
  input while its window is unfocused; `--focus always` exists for those few
  games.
* The view turns but nothing moves — read the whitelist printed at startup.
  `hold=[D, 1, 2, ...]` cannot walk forward; re-running `--calibrate` while
  actually playing the game is the fix.
* The view never turns — the calibrated mouse step is too small. A game that
  locks the cursor reports a nearly still cursor to the recorder, so the
  measured step comes out at a pixel or two. `--mouse-turn 20` sets it
  directly.

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
  runtime.py         checkpoints, hotkeys, the start key, Ctrl+C, priority
  cli.py             argument parsing and the mode dispatch
```

## Requirements

Python 3.10+, Windows (capture and input use Win32), and:

```bash
python -m pip install -r requirements.txt
```

CPU only, by design. There is no CUDA path because there is nothing here big
enough to need one.
