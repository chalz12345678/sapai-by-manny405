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
rules say should happen (gold costs, sell value, lives lost), and the abilities
that can explain anything extra. Hard rule breaks such as more than 5 pets or
negative gold are flagged in red. Use **Copy log** or **Download log** to share a
game when reporting a bug.

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
