"""
botcore - a small, CPU-friendly, game-agnostic reinforcement-learning bot.

The design goals, in the order they constrained the code:

1. **Light.**  One small CNN and one GRU cell run per control step.  There is
   no transformer, no audio branch and no giant action table.  A control step
   costs single-digit milliseconds on a laptop CPU, so the bot spends its time
   watching the game rather than thinking about it.

2. **It cannot stall.**  Every loop in the program has a hard bound, and every
   structure that grows has a cap.  A watchdog measures the real per-step and
   per-update cost and says so out loud, and a liveness monitor notices when
   the game has stopped responding to anything and stops pretending otherwise.
   "Runs for a while then stops making progress" is treated as a bug to be
   detected, not a mystery.

3. **Game agnostic.**  The bot sees pixels, presses keys.  There is no
   knowledge of any game inside it: the keys it may use come from a calibrated
   whitelist, and the learning signal is entirely intrinsic.  Point it at a
   different window and it starts over with no code changes.

4. **It actually learns.**  Intrinsic reward is shaped so that the profitable
   behaviour is "reach states you have never reached before", which is the
   generic version of making progress in a game.  That is verifiable: see
   ``--selftest``, which trains the identical algorithm on a synthetic game
   whose progress is known, and reports whether the score went up.
"""

__version__ = "2.0"

__all__ = [
    "config",
    "capture",
    "keys",
    "model",
    "novelty",
    "policy",
    "trainer",
    "session",
    "replay",
    "synth",
]
