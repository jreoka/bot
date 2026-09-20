#!/usr/bin/env python3
"""
bot1.py - a small, CPU-friendly, game-agnostic bot that learns to play a game
from pixels, with no knowledge of the game.

    python bot1.py                 train (or continue training) on the game
    python bot1.py --selftest      prove it learns, on a game it has never seen
    python bot1.py --calibrate     show it which keys you use (run this once)
    python bot1.py --benchmark     is this machine fast enough?
    python bot1.py --list-windows  find the handle for --window
    python bot1.py --help          everything else

Training starts idle: click the game window, then press F8 to begin sending
input, F8 again to pause, F9 to save, F10 to quit. Input is only ever sent while
the game window has focus, so the terminal stays usable.

The implementation lives in the `botcore` package next to this file. The short
version of how it works:

  * It looks at the game window (grey frames plus a motion channel) and presses
    keys through a calibrated whitelist. No audio, no game-specific code, and no
    reading of a score.
  * The policy is a small CNN plus a GRU - about 400k parameters - so a decision
    costs a few milliseconds on a laptop CPU and the model never fights the game
    for the machine.
  * The reward is intrinsic: reaching states the bot has never reached since the
    last reset, with a depth term so that pressing on is worth more than
    standing still. Resets are inferred from the screen, because a death or a
    respawn is not visible to a bot that knows nothing about the game.
  * Every loop has a hard budget and every memory has a cap, and the session
    prints a specific diagnosis when something has gone wrong - a frozen
    window, a policy that stopped pressing anything, an update that has grown
    too slow. "It ran for a while and then stopped doing anything" is treated
    as a bug to be detected and reported, not a mystery.

Run `python bot1.py --selftest` first if you want to see it learn before
pointing it at a game.
"""

from __future__ import annotations

import os
import sys

# Running this file directly puts its own directory on sys.path, so the package
# next to it imports without installation. Running it from elsewhere (or through
# a symlink) needs the same guarantee.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from botcore.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
