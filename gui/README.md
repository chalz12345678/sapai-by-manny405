# sapai arena (GUI)

A browser GUI for the engine: play against the trained models, or pick two
agents and watch them play each other.

```bash
python -m gui.server          # open http://127.0.0.1:8405
```

Only numpy is needed. The trained MaskablePPO models in `models/` run through a
numpy port of their actor network (`gui/npmodel.py`), so torch, gymnasium and
sb3-contrib are not required to play.

## Modes

- **Play**: you shop and an agent shops; ending the turn plays the battle,
  replayed step by step with a log of every attack and ability.
- **Watch models**: choose a left and right agent, then step through rounds or
  auto play. Each side's shop actions for the turn are listed under its team.
  **Run series** plays 10 to 100 full games and reports the win rates.

Agents: every model in `gui/models/` (the token transformer and MLP checkpoints
with their benchmark ratings), the greedy bot and a random bot.

## Sound

Sound effects for buying, selling, rolling, freezing, feeding, stacking,
leveling up, battle hits, faints and results. They are synthesized in the
browser with the Web Audio API, so there are no audio files. Use the speaker
button and volume slider at the top; the setting is remembered.

## Battle replays

Battles replay one engine phase at a time: start of battle abilities, each
attack, hurt and faint effects, knock-outs, summons. A pet that takes a fatal
hit stays on the board, greyed out, until it faints. Pick a **Battle speed**
under the replay; **Step** waits for you, and ← / → (or ◀ / ▶) move one action
back or forward. The replay wraps the engine's own phase function rather than
reimplementing battles, so it follows the real rules.

### 3D view (prototype)

Switch a battle to **3D** with the 2D / 3D toggle on the battle panel. Pets are
cardboard standees on a small 3D meadow, and every replay step is animated
from the step's events: front pets lunge and collide on attacks, an ability
makes its pet glow and sends an orb to each target, damage flashes red with a
number, buffs sparkle, fainted pets tip over and sink away, and summons pop
up face down and flip over. The camera pushes in on each attack and ability and
eases back to the wide shot, and the standees sway a little between actions. It follows the same step controls and battle speed as the 2D view. It uses
Three.js r128, loaded from cdnjs the first time you open the 3D view, so it
needs an internet connection then; everything else works offline.

## Talking to the model

Press **Talk to the model** to ask the trained model about its play. The models
are policy networks and can't really talk, so answers come from their real
numbers: the probability it gave each move it made and the moves it nearly
made (recorded at every decision), and its critic's value estimate of how the
game is going. Try "why did you sell the duck?", "recap your last turn", "how
confident are you?", "rate my team" or "what would you do in my spot?".

For free-form chat, install the Anthropic SDK and set an API key before
starting the server. Claude then speaks for the model, given only those numbers
and the game state, and is told not to invent reasons beyond them:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...        # PowerShell: $env:ANTHROPIC_API_KEY="..."
python -m gui.server
```

It uses `claude-opus-5-5` at low effort by default; set `SAPAI_CHAT_MODEL` to
use another model.

## Music and effects

Two original tunes play live in the browser (Web Audio, no audio files): a calm
shop theme and a faster battle theme, crossfading between them. The ♪ button
and slider control music separately from sound effects. Shop clicks, purchases,
sales, rolls, feeding, freezing and level ups have particle effects, and battles
show damage numbers and faint puffs on a painted meadow backdrop (night sky in
dark mode). All effects are drawn on canvas and turn off with the system's
reduce-motion setting.

## Saved games

Every game is saved to `gui/saves/<id>.json` after each move (the 100 most
recent are kept), and the page remembers which game it was showing. Refreshing
the page, or restarting the server, picks up where you left off.

## Game log

The **Game log** panel records every shop action, turn start, end of turn and
battle: what changed (gold, team and shop stats, levels, perks), what the basic
rules say should happen (gold costs, sell value, lives lost), and every ability
or perk that fired, with exactly whose stats it changed, for example
`effect: Fish (L1): Level-up: Give two friends +1 attack and +1 health. → Ant
2/2 → 3/3 (+1/+1); Otter 1/4 → 2/5 (+1/+1)`. Hard rule breaks such as more than 5 pets or
negative gold are flagged in red. Use **Copy log** or **Download log** to share a
game when reporting a bug.

## Replays

Every game you play or watch (not series) is kept in `gui/replays/`, one
compressed file per game plus an index, updated after every round. The
**Replays** tab lists them newest first, with search and filters (your games,
watched matches, games you won, finished or unfinished). Open one to rename it,
add a note, and step through any turn: both sides' shop moves, the models'
confidence for each move, the battle step by step in 2D or 3D, and that turn's
game log. **View replay** on the game-over banner opens the game you just
finished. Games saved before replays existed are imported on startup from
their game log, so their battles show as text.

## Battle Lab

The **Battle Lab** tab recreates battles from your own Super Auto Pets game
and checks the engine against them. It runs entirely on your computer with
numpy: no API key, nothing sent anywhere.

1. **Share the game window** (desktop browsers). The lab keeps one frame a
   second for the last 30 seconds in a strip. Play as usual; after a battle,
   switch to the browser and click the frame where the battle starts, then
   the one where it ends. You can also load screenshots instead.
2. **Set up the screen layout** once, on a start-of-battle frame. Choose
   **Automatically (from the hearts)** and press Start: the lab finds every
   pet's red health heart and the attack stone beside it, fits the regular
   spacing between pets (so missing or empty slots are filled in), and
   places every box from that. Hats are ignored: the pet is told apart by
   the part of its picture below where hats sit, and the level by its
   number, which hats don't cover. The opponent's pets are flipped to face
   the same way as yours, so a pet learned on one side is known on the
   other. Perks sit in different places on different pets, so the perk box
   only covers where they usually are; check perks by eye.
   Setting up by hand still works (**Everything, by hand**): click the 5
   pet positions on each side (front pet first), then drag boxes around
   your front pet's picture, attack number, health number, perk spot and
   level label. Any part can be redone on its own.
3. **Read both frames.** The lab fills in each side's pets, stats, levels and
   perks from what it has learned. Guesses it isn't sure of are highlighted.
   It knows nothing at first: type the teams in and press **Learn from my
   corrections**, and it keeps pictures of each pet, digit, perk and level
   it saw. Each battle you correct teaches it more.
4. **Recreate battle.** The engine plays the battle in the battle viewer and
   runs it 200 times to cover random abilities. With the end frame filled in,
   it says whether the engine can end the battle the way the game did.
5. **Save as a test.** Battles that match go into
   `tests/scenarios/from_game.yaml` (confirmed rules). Ones the engine can't
   reproduce go into `tests/scenarios/game_discrepancies.yaml`: each is a
   rule to look into, and the test suite expects them to fail until the
   engine is fixed.

What the lab has learned is kept in `gui/lab_data/` (not in git). Delete that
folder to start over, for example after the game changes its art.

## Move suggestions overlay

A small window that sits in the top-left corner of your screen, reads the
Super Auto Pets shop and shows the model's top 3 moves, each with its
confidence (the model's probability for that move among all legal moves).
It only suggests; you make the moves.

```
pip install pillow
python -m gui.server          # for setting it up and teaching it
python -m gui.overlay         # the overlay
```

On a Mac with Homebrew Python, Tk may need installing first:
`brew install python-tk`. macOS also asks to let Terminal record the screen
(System Settings > Privacy & Security > Screen Recording); allow it and
start the overlay again. Play with the game in a window (maximized is
fine), not macOS full screen, where other windows can't stay on top.

Teaching it, the first time and whenever it is unsure (it lists what it is
unsure about):

1. On the shop screen, press **Read now** in the overlay. It saves what it
   saw.
2. Press **Set up**, which opens the **Shop Reader** tab, and press **Load
   the overlay's last capture**.
3. The first time only, **Set up layout > Everything**: click your 5 team
   pets (front first), every shop slot left to right (then **Done with shop
   slots**), and drag boxes around the parts it reads. Any part can be
   redone on its own later.
4. Correct the tables and press **Learn from my corrections** (**Undo last
   lesson** takes back a mistake). **Suggest moves** shows what the chosen
   model would do with the table as it is.

The overlay picks up new lessons straight away. Pick the model in its
drop-down; `--every 2` reads every 2 seconds, and untick **Auto** to read
only when you press **Read now**.

## Updating the models

`gui/models/*.npz` are the actor weights exported from `models/*.zip`. After
training a new checkpoint, copy it to `models/` and re-export:

```bash
python -m gui.export_models
```

The exporter reads the SB3 checkpoint directly, so it also works without torch.
A numpy model plays like `MaskablePPO.predict(..., deterministic=True)`: the
highest scoring legal action.

## Your own agent

Write a file with a `shop(seat, rng)` function and pass it to the server:

```python
"""Buys the first pet it can afford."""
def shop(seat, rng):
    for a in np.flatnonzero(seat.mask()):
        name, args = E.ACTIONS[a]
        if name == "buy_pet":
            seat.act(int(a))
            return
```

```bash
python -m gui.server --agent my_agent.py
```

`seat.player` is the `sapai.player.Player`, `seat.mask()` the legal action mask,
`seat.obs()` the model observation and `seat.act(a)` applies an action from
`sapai.rl.env.ACTIONS`. `E` (`sapai.rl.env`) and `np` are available in the file.
Return before ending the turn; the match ends it for you.
