# Battle scenarios

Each file here is a list of battles with the result the real game gives.
They check the engine against the game: `python -m sapai.scenarios` runs them
all, and so does the test suite (`tests/test_scenarios.py`).

## Adding a battle from the real game

When a battle in the game surprises you (or the engine), write it down here:

```yaml
- name: Short description of what it shows
  source: real game, 2026-11-14, turn 7 replay      # or ability text
  verified: true                                    # true = seen in the game
  team: [Hedgehog 3/2, Ant 2/1 L2 Garlic]           # your team, front first
  enemy: [Fish 2/3, Beaver 3/2]                     # the other team, front first
  result: win                                       # win, loss or draw (your view)
  survivors:                                        # optional: who is left
    team: [Ant 3/2 L2 Garlic]
    enemy: []
```

- **Pets**: `Name attack/health`, then optionally `L2` or `L3` and a perk
  (`Garlic`, `Melon`, `Meat Bone`, `Steak`, `Honey`, `Mushroom`, `Chili`,
  `Coconut`, `Peanut`, `Cake`, `Bread`, `Weak`). Use the stats shown at the
  start of the battle, before any start-of-battle abilities.
- **result** and **survivors** are both optional; give whichever you know.
  Survivors are checked with their stats at the end of the battle.
- **Random battles**: if the battle depends on chance (for example Mosquito
  picking a target), give `odds` instead, like `odds: {win: 0.5, draw: 0.5}`.
  The runner plays it `runs` times (400 by default) and allows a margin of
  `tolerance` (0.08 by default). A scenario with an exact result that turns
  out to be random fails, so hidden randomness shows up.
- **verified**: `false` means the expected result was worked out from the
  rules or ability text and has not been seen in the game yet. Confirming
  those in the game is the most useful thing to do with this folder.

If a value contains a colon (for example quoting ability text), put it in
quotes: `source: 'Mosquito text "Start of battle: ..."'`.

Screenshots or a screen recording of the battle are the easiest way to get
the starting stats right.
