"""
Match controller for the sapai GUI (``gui/index.html``).

A match has two seats. Each seat is the human or an agent: the greedy bot, the
random bot, one of the trained models in ``models/`` (run with numpy through
``gui/npmodel.py``), or your own Python agent. So the same controller covers
"you vs a model" and "model vs model".

All shop actions go through ``sapai.rl.env`` (``action_mask``, ``apply_action``
and ``encode``), the exact action and observation layout the models were
trained on. The game rules match ``SAPEnv``: 10 lives, 10 wins to win, lives
lost per turn from ``data["turns"]``.

Works from a normal shell too::

    from gui.game import Match
    m = Match("token_16.8M", "greedy")
    while not m.over: m.play_round()
"""

import json
import random
import sys
import types

import numpy as np


def _stub_missing():
    """sapai.rl.env imports gymnasium only for the Env base class and spaces.
    Stub them when absent (the browser has neither gymnasium nor torch)."""
    try:
        import gymnasium  # noqa: F401
    except Exception:
        gym = types.ModuleType("gymnasium")
        spaces = types.ModuleType("gymnasium.spaces")

        class Env:
            def reset(self, *a, **k):
                pass

        spaces.Box = lambda *a, **k: None
        spaces.Discrete = lambda *a, **k: None
        gym.Env = Env
        gym.spaces = spaces
        sys.modules["gymnasium"] = gym
        sys.modules["gymnasium.spaces"] = spaces


_stub_missing()

from sapai.battle import Battle  # noqa: E402
from sapai.cli import Opponent as GreedyShopper, log_line  # noqa: E402
from sapai.data import data  # noqa: E402
from sapai.player import Player  # noqa: E402
from sapai.rl import env as E  # noqa: E402
from sapai.rl.opponents import RandomBot  # noqa: E402

MAX_WINS = 10
START_LIVES = 10
MAX_ACTIONS = 30


# ------------------------------------------------------------- formatting
def _short(name):
    return name.split("-", 1)[1] if "-" in name else name


FALLBACK_EMOJI = {
    "pet-pigeon": "\U0001F426", "pet-armadillo": "\U0001F994", "pet-wolverine": "\U0001F9A1",
    "food-cake": "\U0001F370", "food-bread": "\U0001F35E", "food-bread-crumbs": "\U0001F35E",
    "food-better-apple": "\U0001F34E", "food-best-apple": "\U0001F34E",
    "food-better-milk": "\U0001F95B", "food-best-milk": "\U0001F95B",
    "status-cake": "\U0001F370", "status-bread": "\U0001F35E",
}


def _emoji(name):
    group = "pets" if name.startswith("pet-") else "foods"
    try:
        return data[group][name]["image"]["unicodeCodePoint"]
    except Exception:
        return FALLBACK_EMOJI.get(name, "\U0001F43E")


def _ability(name, level=1):
    if name.startswith("pet-"):
        e = data["pets"].get(name, {})
        ab = e.get(f"level{level}Ability") or e.get("level1Ability") or {}
    else:
        ab = data["foods"].get(name, {}).get("ability") or {}
    return ab.get("description", "")


def _status(status):
    if status in (None, "none"):
        return None
    e = data["statuses"].get(status, {})
    return {"name": e.get("name", _short(status)),
            "emoji": (e.get("image") or {}).get("unicodeCodePoint") or FALLBACK_EMOJI.get(status, ""),
            "desc": (e.get("ability") or {}).get("description", "")}


def pet_dict(pet):
    if pet is None or pet.name == "pet-none":
        return None
    return {"name": _short(pet.name).replace("-", " "), "emoji": _emoji(pet.name),
            "attack": int(pet.attack), "health": int(pet.health),
            "level": int(pet.level), "xp": int(pet.experience),
            "tier": data["pets"].get(pet.name, {}).get("tier", 1),
            "status": _status(pet.status), "desc": _ability(pet.name, int(pet.level))}


def shop_dict(slot):
    o = slot.obj
    if slot.slot_type == "food":
        d = {"kind": "food", "name": _short(o.name).replace("-", " "), "emoji": _emoji(o.name),
             "desc": _ability(o.name), "targeted": E.food_needs_target(o)}
    else:
        d = pet_dict(o)
        d["kind"] = slot.slot_type
    d["cost"] = int(slot.cost)
    d["frozen"] = bool(slot.frozen)
    return d


def lives_lost(turn):
    return data["turns"][f"turn-{min(turn, 11)}"]["livesLost"]


def describe_action(p, a):
    """Human readable text for env action a, before it is applied to p."""
    name, args = E.ACTIONS[a]
    team = E.team_pets(p)
    shop = p.shop.slots

    def sn(i):
        return _short(shop[i].obj.name).replace("-", " ")

    def tn(j):
        return _short(team[j].name).replace("-", " ")

    try:
        return {
            "end_turn": lambda: "ended turn",
            "roll": lambda: "rolled",
            "buy_pet": lambda: f"bought {sn(args[0])}",
            "buy_combine": lambda: f"bought {sn(args[0])} onto {tn(args[1])}",
            "buy_food": lambda: f"fed {sn(args[0])} to {tn(args[1])}",
            "buy_food_untargeted": lambda: f"bought {sn(args[0])}",
            "sell": lambda: f"sold {tn(args[0])}",
            "freeze": lambda: ("unfroze " if shop[args[0]].frozen else "froze ") + sn(args[0]),
            "swap": lambda: f"swapped {tn(args[0])} and {tn(args[1])}",
            "combine": lambda: f"combined {tn(args[0])} with {tn(args[1])}",
        }[name]()
    except Exception:
        return name


# ------------------------------------------------------------------ agents
class Seat:
    """One side of a match: its Player plus what the models observe."""

    def __init__(self, kind):
        self.kind = kind
        self.player = Player(lives=START_LIVES)
        self.wins = 0
        self.last_result = None  # this seat's view: 0 won, 1 lost, 2 draw
        self.actions = 0
        self.log = []

    def obs(self):
        return E.encode(self.player, self.actions, self.last_result, 25, self.wins)

    def mask(self):
        return E.action_mask(self.player)

    def act(self, a):
        """Apply env action a (not end_turn). Returns the log text."""
        text = describe_action(self.player, a)
        E.apply_action(self.player, a)
        self.actions += 1
        self.log.append(text)
        return text


AGENTS = {}


def register_agent(agent_id, label, desc, group, shop):
    """shop(seat, rng) plays one shop phase for a Seat (stop before end turn)."""
    AGENTS[agent_id] = {"id": agent_id, "label": label, "desc": desc,
                        "group": group, "shop": shop}


def _greedy(seat, rng):
    bot = GreedyShopper(rng)
    bot.player = seat.player
    before = len(seat.player.action_history)
    bot.shop_turn()
    for entry in seat.player.action_history[before:]:
        seat.log.append(_history_text(entry))


def _history_text(entry):
    """Log text for one Player.action_history entry made by the greedy bot."""
    try:
        name, args = entry[0], entry[1]
        names = [_short(str(x["name"])).replace("-", " ")
                 for x in args if isinstance(x, dict) and "name" in x]
        verb = {"buy_pet": "bought", "buy_combine": "bought and stacked",
                "buy_food": "bought", "roll": "rolled", "sell": "sold",
                "combine": "combined"}.get(name, name.replace("_", " "))
        if name == "buy_food" and len(names) > 1:
            return f"fed {names[0]} to {', '.join(names[1:])}"
        return " ".join([verb] + names[:1])
    except Exception:
        return "acted"


def _random(seat, rng):
    for _ in range(15):
        if rng.random() < 0.15:
            break
        legal = [a for a in np.flatnonzero(seat.mask()) if a != E.END_TURN]
        if not legal:
            break
        try:
            seat.act(int(rng.choice(legal)))
        except Exception:
            pass


def model_shop(policy):
    def shop(seat, rng):
        for _ in range(MAX_ACTIONS):
            a, _ = policy.predict(seat.obs(), seat.mask())
            if a == E.END_TURN:
                break
            try:
                seat.act(a)
            except Exception:
                seat.actions += 1
    return shop


register_agent("greedy", "Greedy bot", "Merges duplicates, buys the highest tier pets, feeds food. Rated 1000.",
               "bot", _greedy)
register_agent("random", "Random bot", "Takes random legal shop actions.", "bot", _random)

LAYOUT = {
    "MAX_TEAM": E.MAX_TEAM, "MAX_SHOP": E.MAX_SHOP, "TEAM_TOKEN": E.TEAM_TOKEN,
    "SHOP_TOKEN": E.SHOP_TOKEN, "GLOBAL_TOKEN": E.GLOBAL_TOKEN,
    "TEAM_PRESENT": E.N_PETS + E.N_STATUSES, "N_SHOP_KINDS": len(E.SHOP_KINDS),
    "PAIR_A": [a for a, b in E.PAIRS], "PAIR_B": [b for a, b in E.PAIRS],
}

MODEL_INFO = {
    "token_16.8M": ("Token net 16.8M", 1470), "token_13.5M": ("Token net 13.5M", 1459),
    "token_11.8M": ("Token net 11.8M", 1440), "token_6.3M": ("Token net 6.3M", 1427),
    "token_3.5M": ("Token net 3.5M", 1393), "mlp_19.3M": ("MLP 19.3M", 1420),
    "mlp_10.9M": ("MLP 10.9M", 1332), "mlp_8.2M": ("MLP 8.2M", 1312),
}


def load_model(name, path):
    """Register a trained model from an exported .npz file."""
    from gui.npmodel import NumpyPolicy

    with np.load(path) as f:
        policy = NumpyPolicy(dict(f), layout=LAYOUT)
    label, rating = MODEL_INFO.get(name, (name, None))
    kind = "transformer over team and shop tokens" if policy.kind == "token" else "256x256 MLP"
    steps = name.split("_")[-1]
    desc = f"MaskablePPO, {kind}, {steps} training steps" + (f". Rated {rating}." if rating else ".")
    register_agent(name, label, desc, "model", model_shop(policy))
    AGENTS[name]["rating"] = rating
    return name


def load_agent_source(agent_id, source):
    """Register an agent from Python source defining shop(seat, rng)."""
    mod = types.ModuleType(f"user_agent_{agent_id}")
    mod.__dict__.update({"E": E, "np": np})
    exec(compile(source, f"<{agent_id}>", "exec"), mod.__dict__)
    if not callable(mod.__dict__.get("shop")):
        raise Exception("Your agent needs a function shop(seat, rng).")
    desc = (mod.__doc__ or "Your agent.").strip().splitlines()[0]
    register_agent(agent_id, agent_id, desc, "custom", mod.shop)
    return json.dumps(agent_list()[-1])


def agent_list():
    return [{k: v for k, v in a.items() if k != "shop"} for a in AGENTS.values()]


def agents_json():
    return json.dumps(agent_list())


# ------------------------------------------------------------------ battle
def _snap(b, label, lines=()):
    return {"label": label, "lines": list(lines),
            "teams": [[pet_dict(s.pet) for s in t if not s.empty and s.pet.health > 0]
                      for t in (b.t0, b.t1)]}


def _rename(line, names):
    a, b = names
    if a == "You":
        return line
    return (line.replace("Your ", f"{a}'s ").replace(" their ", f" {b}'s ")
            .replace("yours takes", f"{a}'s takes").replace("theirs takes", f"{b}'s takes")
            .replace("You:", f"{a}:").replace("Opponent:", f"{b}:"))


def _lines(entries_by_phase, names):
    out = []
    for phase, entries in entries_by_phase.items():
        if phase.startswith("phase_move"):
            continue
        for entry in entries:
            try:
                line = log_line(entry)
            except Exception:
                line = None
            if line:
                out.append(_rename(line, names))
    return out


def replay_battle(t0, t1, names=("You", "Opponent")):
    b = Battle(t0, t1)
    frames = [_snap(b, "Lineup")]
    b.start()
    frames.append(_snap(b, "Start of battle", _lines(b.battle_history.get("start", {}), names)))
    for i in range(200):
        b.pet_priority = b.update_pet_priority(b.t0, b.t1)
        more = b.attack(i)
        frames.append(_snap(b, f"Attack {i + 1}",
                            _lines(b.battle_history.get(f"attack {i}", {}), names)))
        if not more:
            break
    return frames, int(b.check_battle_result())


# ------------------------------------------------------------------- match
class Match:
    def __init__(self, seat0="human", seat1="token_16.8M", seed=None):
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)
        self.seats = [Seat(seat0), Seat(seat1)]
        self.over = False
        self.result = None
        self.history = []

    def label(self, i):
        k = self.seats[i].kind
        if k == "human":
            return "You"
        lab = AGENTS.get(k, {}).get("label", k)
        if self.seats[0].kind == self.seats[1].kind:
            lab += " (A)" if i == 0 else " (B)"
        return lab

    def _seat_state(self, i):
        s = self.seats[i]
        p = s.player
        legal = s.mask()
        return {"kind": s.kind, "label": self.label(i), "turn": int(p.turn),
                "gold": int(p.gold), "lives": int(p.lives), "wins": int(s.wins),
                "team": [pet_dict(x) for x in E.team_pets(p)],
                "shop": [dict(shop_dict(x), index=i) for i, x in enumerate(p.shop.slots[:E.MAX_SHOP])
                         if not x.empty],
                "legal": [int(a) for a in np.flatnonzero(legal)], "log": s.log}

    def state(self):
        return {"seats": [self._seat_state(0), self._seat_state(1)], "over": self.over,
                "result": self.result, "history": self.history, "maxWins": MAX_WINS,
                "turn": int(self.seats[0].player.turn),
                "nextDamage": lives_lost(self.seats[0].player.turn)}

    def state_json(self):
        return json.dumps(self.state())

    def action_index(self, name, *args):
        start = E.ACTION_START[name]
        if name in ("buy_combine", "buy_food"):
            return start + int(args[0]) * E.MAX_TEAM + int(args[1])
        if name in ("swap", "combine"):
            a, b = sorted(int(x) for x in args)
            return start + E.PAIRS.index((a, b))
        if args:
            return start + int(args[0])
        return start

    def do(self, name, *args):
        """Human action on seat 0 by env action name, e.g. do("buy_pet", 2)."""
        err, battle = None, None
        try:
            if self.over:
                raise Exception("The game is over. Start a new one.")
            if name == "end_turn":
                battle = self.play_round()
            else:
                if self.seats[0].kind != "human":
                    raise Exception("No human seat in this match.")
                a = self.action_index(name, *args)
                if not self.seats[0].mask()[a]:
                    raise Exception(_why_illegal(self.seats[0].player, name))
                self.seats[0].act(a)
        except Exception as e:
            err = str(e)
        return json.dumps({"ok": err is None, "error": err, "state": self.state(),
                           "battle": battle})

    def play_round(self):
        for i, s in enumerate(self.seats):
            if s.kind != "human":
                try:
                    AGENTS[s.kind]["shop"](s, self.rng)
                except Exception as e:
                    s.log.append(f"agent error: {e}")
        logs = [list(s.log) for s in self.seats]
        turn = int(self.seats[0].player.turn)
        for s in self.seats:
            s.player.end_turn()
            s.player.team.move_forward()
        try:
            frames, w = replay_battle(self.seats[0].player.team, self.seats[1].player.team,
                                      (self.label(0), self.label(1)))
        except Exception as e:
            frames, w = [{"label": f"Engine error: {e}", "lines": [], "teams": [[], []]}], 2
        lost = lives_lost(turn)
        views = {0: (0, 1), 1: (1, 0), 2: (2, 2)}[w]
        for s, r in zip(self.seats, views):
            s.last_result = r
            if r == 0:
                s.wins += 1
            elif r == 1:
                s.player.lives -= lost
        self.history.append(["win", "loss", "draw"][w])
        a, b = self.seats
        if a.wins >= MAX_WINS or b.player.lives <= 0:
            self.over, self.result = True, "seat0"
        elif b.wins >= MAX_WINS or a.player.lives <= 0:
            self.over, self.result = True, "seat1"
        else:
            for s in self.seats:
                s.player.start_turn(winner=(s.last_result != 1))
                s.actions = 0
                s.log = []
        return {"frames": frames, "outcome": ["win", "loss", "draw"][w], "lives": lost,
                "turn": turn, "shopLogs": logs}

    def step_json(self):
        if self.over:
            return json.dumps({"ok": False, "error": "The match is over.", "state": self.state()})
        b = self.play_round()
        return json.dumps({"ok": True, "error": None, "state": self.state(), "battle": b})


def _why_illegal(p, name):
    if name.startswith("buy") and p.gold < 3:
        return f"Not enough gold ({p.gold})."
    if name == "buy_pet" and len(p.team) >= E.MAX_TEAM:
        return "Your team is full. Sell a pet or drop the new one onto a matching pet."
    if name == "roll":
        return "Rolling costs 1 gold."
    if name == "buy_food":
        return "Drop this food onto one of your pets."
    if name == "buy_combine":
        return "You can only stack a pet onto the same pet below level 3."
    if name == "combine":
        return "Only matching pets below level 3 can be combined."
    return "That move is not allowed right now."
