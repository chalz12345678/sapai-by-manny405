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
import re
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
from sapai.cli import Opponent as GreedyShopper, log_line, pet_parts  # noqa: E402
from sapai.data import data  # noqa: E402
from sapai.player import Player  # noqa: E402
from sapai.rl import env as E  # noqa: E402
from sapai.rl.opponents import RandomBot  # noqa: E402, F401
from gui import audit  # noqa: E402

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


### Plain explanations of ability trigger words, for new players
TRIGGERS = {
    "Faint": "Happens when this pet dies (its health drops to 0).",
    "Start of battle": "Happens once, before the first attack of each battle.",
    "Start of turn": "Happens when a new shop turn begins.",
    "End turn": "Happens when you press End turn, before the battle.",
    "Sell": "Happens when you sell this pet.",
    "Buy": "Happens when you buy this pet from the shop.",
    "Hurt": "Happens each time this pet takes damage and survives.",
    "Level-up": "Happens when this pet reaches a new level.",
    "Knock out": "Happens when this pet's attack makes an enemy faint.",
    "Before attack": "Happens just before this pet attacks.",
    "After attack": "Happens just after this pet attacks.",
    "Summoned": "Happens when this pet is summoned onto the team during a battle.",
    "Friend summoned": "Happens when another pet joins your team (bought or summoned).",
    "Friend faints": "Happens when one of your other pets faints.",
    "Friend ahead faints": "Happens when the pet directly in front of this one faints.",
    "Friend ahead attacks": "Happens when the pet directly in front of this one attacks.",
    "Friend hurt": "Happens when one of your other pets takes damage.",
    "Friendly ate food": "Happens when any of your pets eats food from the shop.",
    "Eats food": "Happens when this pet eats food from the shop.",
    "Buy tier 1 pet": "Happens when you buy a tier 1 pet from the shop.",
    "Passive": "Always active. It does not need a trigger.",
}

TERMS = [
    ("until end of battle", "until end of battle",
     "The bonus lasts for the next battle only, then goes away."),
    (r"\bsummon", "Summon", "Adds a new pet to a team during the battle, if that team has room (5 pets max)."),
    (r"\bexperience", "Experience",
     "Pets level up from experience: 2 to reach level 2, then 3 more for level 3."),
    (r"\bahead\b", "Ahead", "Ahead means closer to the front of the line, which fights first."),
    (r"\bbehind\b", "Behind", "Behind means further from the front of the line."),
    (r"works \d+ times?", "Works N times", "The ability stops after that many uses in one turn or battle."),
    (r"triggers \d+ times?", "Triggers N times", "The ability stops after that many uses in one battle."),
    (r"\btier\b", "Tier", "Pets come in tiers 1 to 6. Higher tiers show up in the shop on later turns."),
    (r"\bknocks? out\b", "Knock out", "Knocking out a pet means making it faint."),
]


def _perk_by_name():
    out = {}
    for sid, e in data["statuses"].items():
        name = e.get("name")
        if name:
            out[name] = sid
    return out


PERKS = _perk_by_name()


def glossary(text):
    """Explanations for the game words used in an ability or food text."""
    out, seen = [], set()
    if not text:
        return out
    head = text.split(":", 1)[0] if ":" in text else None
    if head in TRIGGERS:
        out.append({"term": head, "text": TRIGGERS[head]})
        seen.add(head)
    low = text.lower()
    for name, expl in TRIGGERS.items():
        if name not in seen and re.search(re.escape(name.lower()) + r" abilit", low):
            out.append({"term": name, "text": expl})
            seen.add(name)
    for name, sid in PERKS.items():
        if re.search(r"\b" + re.escape(name.lower()) + r"\b", low):
            st = _status(sid)
            out.append({"term": f"{st['emoji']} {name} perk", "text": st["desc"]})
    for key, term, expl in TERMS:
        if term not in seen and re.search(key, low):
            out.append({"term": term, "text": expl})
            seen.add(term)
    return out


def _levels(name):
    e = data["pets"].get(name, {})
    return [(e.get(f"level{i}Ability") or {}).get("description", "") for i in (1, 2, 3)]


def pet_dict(pet):
    if pet is None or pet.name == "pet-none":
        return None
    level = int(pet.level)
    desc = _ability(pet.name, level)
    st = _status(pet.status)
    gloss = glossary(desc)
    try:
        sell = int(pet.sell_value)
    except Exception:
        sell = level
    return {"name": _short(pet.name).replace("-", " "), "emoji": _emoji(pet.name),
            "attack": int(pet.attack), "health": int(pet.health),
            "level": level, "xp": int(pet.experience),
            "tier": data["pets"].get(pet.name, {}).get("tier", 1),
            "status": st, "desc": desc, "levels": _levels(pet.name),
            "glossary": gloss, "sell": sell}


def shop_dict(slot):
    o = slot.obj
    if slot.slot_type == "food":
        desc = _ability(o.name)
        perk_id = (data["foods"].get(o.name, {}).get("ability") or {}).get("effect", {}).get("status")
        perk = _status(perk_id) if perk_id else None
        gloss = [g for g in glossary(desc) if not (perk and g["term"].endswith(perk["name"] + " perk"))]
        d = {"kind": "food", "name": _short(o.name).replace("-", " "), "emoji": _emoji(o.name),
             "desc": desc, "targeted": E.food_needs_target(o), "perk": perk,
             "tier": data["foods"].get(o.name, {}).get("tier", 1), "glossary": gloss}
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
        self.recorder = None  # set by Match: recorder(seat, title, before, after, ...)
        self.thoughts = []  # trained models: what they weighed at each decision

    def obs(self):
        return E.encode(self.player, self.actions, self.last_result, 25, self.wins)

    def mask(self):
        return E.action_mask(self.player)

    def act(self, a):
        """Apply env action a (not end_turn). Returns the log text."""
        text = describe_action(self.player, a)
        name, args = E.ACTIONS[a]
        before = audit.snapshot(self.player)
        audit.start_capture()
        try:
            E.apply_action(self.player, a)
        finally:
            effects = audit.stop_capture()
            self.actions += 1
            self.log.append(text)
            if self.recorder:
                self.recorder(self, text, before, audit.snapshot(self.player), name, args, effects)
        return text


AGENTS = {}


def register_agent(agent_id, label, desc, group, shop):
    """shop(seat, rng) plays one shop phase for a Seat (stop before end turn)."""
    AGENTS[agent_id] = {"id": agent_id, "label": label, "desc": desc,
                        "group": group, "shop": shop}


def _greedy(seat, rng):
    """The greedy bot from sapai.cli, one logged action at a time. Mirrors
    Opponent.shop_turn exactly."""
    p = seat.player
    bot = GreedyShopper(rng)
    bot.player = p
    for _ in range(40):
        if p.gold <= 0:
            break
        before = audit.snapshot(p)
        n = len(p.action_history)
        audit.start_capture()
        try:
            acted = bot.take_action()
            if not acted and p.gold >= 1 and rng.random() < 0.5:
                p.roll()
                acted = "roll"
        finally:
            effects = audit.stop_capture()
        if acted is True:
            entry = p.action_history[n] if len(p.action_history) > n else ("acted", [])
            text, name = _history_text(entry), str(entry[0])
            args = _greedy_args(entry, before)
        elif acted == "roll":
            text, name, args = "rolled", "roll", ()
        else:
            break
        seat.log.append(text)
        if seat.recorder:
            seat.recorder(seat, text, before, audit.snapshot(p), name, args, effects)
    p.team.move_forward()


def _greedy_args(entry, before):
    """Shop and team indices of a greedy action, for the log's rule checks."""
    name, args = entry[0], entry[1]
    names = [str(x.get("name")) for x in args if isinstance(x, dict)]
    shop_i = next((x["i"] for x in before["shop"]
                   if names and x["name"] == audit._short(names[0])), 0)
    if name == "buy_combine" and len(names) > 1:
        team_i = next((k for k, x in enumerate(before["team"]) if x["pet"] == names[1]), 0)
        return (shop_i, team_i)
    return (shop_i,)


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


MAX_THOUGHT_TURNS = 12


def think(policy, seat, obs=None, mask=None):
    """What the policy makes of the seat's position: the move it picks, its
    probability for each legal move, and the critic's value estimate."""
    obs = seat.obs() if obs is None else obs
    mask = seat.mask() if mask is None else mask
    a, z = policy.predict(obs, mask)
    legal = np.flatnonzero(mask)
    zz = z[legal] - z[legal].max()
    pr = np.exp(zz) / np.exp(zz).sum()
    order = np.argsort(-pr)
    names = {int(k): describe_action(seat.player, int(k)) for k in legal}
    counts = {}
    for v in names.values():
        counts[v] = counts.get(v, 0) + 1
    for k, v in names.items():
        args = E.ACTIONS[k][1]
        if counts[v] > 1 and args and E.ACTIONS[k][0] in ("buy_pet", "buy_combine", "buy_food", "freeze"):
            names[k] = f"{v} (shop slot {args[0] + 1})"
    options = [[names[int(legal[k])], round(float(pr[k]), 4)] for k in order[:5]]
    chosen_p = float(pr[list(legal).index(a)])
    return {"choice": names[int(a)], "action": int(a), "confidence": round(chosen_p, 4),
            "options": options, "value": policy.value(obs), "legal_moves": int(len(legal))}


def model_shop(policy):
    def shop(seat, rng):
        turn = int(seat.player.turn)
        for step in range(MAX_ACTIONS):
            t = think(policy, seat)
            t.update(turn=turn, step=step + 1, gold=int(seat.player.gold))
            seat.thoughts.append(t)
            a = t["action"]
            if a == E.END_TURN:
                break
            try:
                seat.act(a)
            except Exception:
                seat.actions += 1
        oldest = turn - MAX_THOUGHT_TURNS
        seat.thoughts[:] = [x for x in seat.thoughts if x["turn"] > oldest]
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
    AGENTS[name]["policy"] = policy
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
    return [{k: v for k, v in a.items() if k not in ("shop", "policy")} for a in AGENTS.values()]


def agents_json():
    return json.dumps(agent_list())


# ------------------------------------------------------------------ battle
def _snap(b, label, lines=()):
    """Board snapshot. A pet at 0 health stays on the board, marked dead,
    until the engine's faint phase removes it, so the hit that killed it
    shows."""
    teams = []
    for t in (b.t0, b.t1):
        row = []
        for s in t:
            if s.empty:
                continue
            d = pet_dict(s.pet)
            if s.pet.health <= 0:
                d["dead"] = True
            row.append(d)
        teams.append(row)
    return {"label": label, "lines": list(lines), "teams": teams}


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
                line = line.replace("deals damage to nothing", "has no enemy left to hit")
                out.append(_rename(line, names))
    return out


PHASE_LABELS = {
    "phase_move_start": "pets move up", "phase_start": "start of battle abilities",
    "phase_hurt_and_faint": "hurt and faint effects", "phase_move_end": "pets move up",
    "phase_attack_before": "before-attack abilities", "phase_hurt_and_faint_ab": "hurt and faint effects",
    "phase_attack": "the front pets attack", "phase_attack_after": "after-attack abilities",
    "phase_hurt_and_faint_aa": "hurt and faint effects", "phase_knockout": "knock-out abilities",
    "phase_hurt_and_faint_k": "hurt and faint effects",
}


def _events(entries):
    """Who did what to whom in one phase, for animating it: kind (Attack,
    Fainted, or the effect function such as DealDamage or ModifyStats), the
    acting pet's side and name, and the names of its targets."""
    out = []
    for entry in entries:
        try:
            if len(entry) < 4 or type(entry[0]) != str:
                continue
            src = pet_parts(entry[2])
            if not src:
                continue
            targets = [p[0].lower() for t in entry[3] for p in pet_parts(t)]
            out.append({"kind": entry[0], "side": int(entry[1][0]), "src": src[0][0].lower(),
                        "targets": targets})
        except Exception:
            continue
    return out


def replay_battle(t0, t1, names=("You", "Opponent")):
    """Run a battle and snapshot the board after every engine phase that did
    something, so the replay can step through each action. The engine's own
    battle_phase is wrapped for the duration of this call, nothing is
    reimplemented, so the replay follows the real rules exactly."""
    import sapai.battle as battle_mod

    b = Battle(t0, t1)
    frames = [_snap(b, "Lineup")]
    frames[0]["phase"] = "lineup"
    where = {"round": "Start of battle"}
    original = battle_mod.battle_phase

    stack = []  # phases currently running, outermost first: [phase, phase_dict, flushed]

    def emit(phase, entries):
        lines = _lines({phase: entries}, names) if not phase.startswith("phase_move") else []
        f = _snap(b, f"{where['round']} · {PHASE_LABELS.get(phase, phase)}", lines)
        f["phase"] = phase
        f["events"] = _events(entries) if not phase.startswith("phase_move") else []
        if lines or f["teams"] != frames[-1]["teams"]:
            frames.append(f)

    def recording_phase(battle_obj, phase, teams, pet_priority, phase_dict):
        if battle_obj is not b:
            return original(battle_obj, phase, teams, pet_priority, phase_dict)
        if stack:
            ### A phase running inside another one (Rhino: each knock-out hit
            ###   runs its own faint phase). Show what the outer phase did so
            ###   far first, so the hit appears before the faint it causes.
            parent = stack[-1]
            done = parent[1].get(parent[0], [])
            emit(parent[0], done[parent[2]:])
            parent[2] = len(done)
        entry = [phase, phase_dict, len(phase_dict.get(phase, []))]
        stack.append(entry)
        try:
            out = original(battle_obj, phase, teams, pet_priority, phase_dict)
        finally:
            stack.pop()
        emit(phase, phase_dict.get(phase, [])[entry[2]:])
        return out

    battle_mod.battle_phase = recording_phase
    try:
        b.start()
        for i in range(200):
            where["round"] = f"Attack {i + 1}"
            b.pet_priority = b.update_pet_priority(b.t0, b.t1)
            if not b.attack(i):
                break
    finally:
        battle_mod.battle_phase = original
    return frames, int(b.check_battle_result())


# ------------------------------------------------------------------- match
class Match:
    def __init__(self, seat0="human", seat1="token_16.8M", seed=None):
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)
        self.seats = [Seat(seat0), Seat(seat1)]
        self.events = []
        self.chat = []
        for s in self.seats:
            s.recorder = self._record_action
        self.over = False
        self.result = None
        self.history = []
        for i, st in enumerate(self.seats):
            snap = audit.snapshot(st.player)
            self._event("start", i, "turn 1 starts", [
                f"gold {snap['gold']}, lives {snap['lives']}",
                "shop now: " + ", ".join(audit._item_str(x) for x in snap["shop"])])

    # ------------------------------------------------------------- game log
    def _event(self, kind, seat, title, lines=(), warn=(), ok=None):
        self.events.append({"n": len(self.events), "turn": int(self.seats[0].player.turn),
                            "kind": kind, "seat": seat,
                            "who": self.label(seat) if seat is not None else "",
                            "title": title, "lines": list(lines), "warn": list(warn),
                            "ok": ok})

    def _record_action(self, seat, text, before, after, name, args, effects=None):
        i = self.seats.index(seat)
        lines = audit.diff(before, after, roll=(name == "roll"))
        if not lines:
            lines = ["nothing changed"]
        exp, notes = audit.expectations(name, list(args), before)
        check, ok = audit.gold_check(exp, before, after)
        fx = audit.effect_lines(effects or [])
        lines += notes + fx
        if not fx:
            lines += audit.ability_notes(name, before, after, args)
        lines.append(check)
        self._event("action", i, text, lines, audit.violations(after), ok)

    def _record_phase(self, kind, title, before, phase, effects=None):
        for i, s in enumerate(self.seats):
            after = audit.snapshot(s.player)
            lines = audit.diff(before[i], after, roll=(kind == "start"))
            if kind == "start":
                lines = [l for l in lines if not l.startswith("gold ")]
                lines.insert(0, f"gold reset to {after['gold']}")
                if any(l.startswith("team:") for l in lines):
                    lines.append("rule: bonuses that last 'until end of battle' wear off here; "
                                 "start of turn abilities, if any, are listed below")
            fx = audit.effect_lines((effects or [[], []])[i])
            notes = fx or audit.ability_notes(phase, before[i], after)
            if kind == "end" and lines == [] and not fx:
                continue
            self._event(kind, i, title, lines + notes, audit.violations(after))

    def events_since(self, n):
        return self.events[max(0, int(n)):]

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
                "nextDamage": lives_lost(self.seats[0].player.turn),
                "logCount": len(self.events)}

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
        before = [audit.snapshot(s.player) for s in self.seats]
        effects = []
        for s in self.seats:
            audit.start_capture()
            try:
                s.player.end_turn()
            finally:
                effects.append(audit.stop_capture())
            s.player.team.move_forward()
        self._record_phase("end", "end of turn effects", before, "end_turn", effects)
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
        teams = [", ".join(f"{p['name']} {p['attack']}/{p['health']}" for p in t) or "no pets"
                 for t in frames[0]["teams"]]
        blines = [f"{self.label(0)} (front first): {teams[0]}",
                  f"{self.label(1)} (front first): {teams[1]}"]
        for f in frames[1:]:
            for line in f["lines"]:
                blines.append(f"{f['label']}: {line}")
        surv = [", ".join(f"{p['name']} {p['attack']}/{p['health']}" for p in t) or "none"
                for t in frames[-1]["teams"]]
        blines.append(f"survivors: {self.label(0)}: {surv[0]} · {self.label(1)}: {surv[1]}")
        res = {0: f"{self.label(0)} won", 1: f"{self.label(1)} won", 2: "draw"}[w]
        if w != 2:
            loser = 1 - w
            blines.append(f"rule: losing on turn {turn} costs {lost} {'life' if lost == 1 else 'lives'}: "
                          f"{self.label(loser)} {self.seats[loser].player.lives + lost} → "
                          f"{self.seats[loser].player.lives}")
        else:
            blines.append("rule: a draw costs no lives")
        warn = []
        if (w == 0 and surv[1] != "none") or (w == 1 and surv[0] != "none"):
            warn.append("the winner's opponent still has pets left")
        self._event("battle", None, f"turn {turn} battle: {res}", blines, warn)
        a, b = self.seats
        if a.wins >= MAX_WINS or b.player.lives <= 0:
            self.over, self.result = True, "seat0"
        elif b.wins >= MAX_WINS or a.player.lives <= 0:
            self.over, self.result = True, "seat1"
        else:
            before = [audit.snapshot(s.player) for s in self.seats]
            effects = []
            for s in self.seats:
                audit.start_capture()
                try:
                    s.player.start_turn(winner=(s.last_result != 1))
                finally:
                    effects.append(audit.stop_capture())
                s.actions = 0
                s.log = []
            self._record_phase("start", f"turn {turn + 1} starts", before, "start_turn", effects)
        return {"frames": frames, "outcome": ["win", "loss", "draw"][w], "lives": lost,
                "turn": turn, "shopLogs": logs}

    # ---------------------------------------------------------- save / load
    def to_save(self):
        """JSON-safe dict of the whole match, for saving to disk."""
        return {"version": 1, "kinds": [x.kind for x in self.seats],
                "seats": [{"player": x.player.state, "wins": x.wins,
                           "last_result": x.last_result, "actions": x.actions, "log": x.log,
                           "thoughts": x.thoughts}
                          for x in self.seats],
                "events": self.events, "history": self.history, "over": self.over,
                "chat": self.chat,
                "result": self.result, "rng": list(self.rng.getstate()[1]),
                "rng_version": self.rng.getstate()[0]}

    @classmethod
    def from_save(cls, d):
        m = cls.__new__(cls)
        m.rng = random.Random()
        m.rng.setstate((d["rng_version"], tuple(d["rng"]), None))
        m.seats = []
        for kind, sd in zip(d["kinds"], d["seats"]):
            seat = Seat(kind)
            seat.player = Player.from_state(sd["player"])
            seat.wins, seat.last_result = sd["wins"], sd["last_result"]
            seat.actions, seat.log = sd["actions"], list(sd["log"])
            seat.thoughts = list(sd.get("thoughts", []))
            seat.recorder = m._record_action
            m.seats.append(seat)
        m.events, m.history = list(d["events"]), list(d["history"])
        m.chat = list(d.get("chat", []))
        m.over, m.result = d["over"], d["result"]
        return m

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
        return "This food doesn't go on one pet. Press Buy to use it." if any(
            s.slot_type == "food" and not s.empty and not E.food_needs_target(s.obj)
            for s in p.shop.slots) else "Drop this food onto one of your pets."
    if name == "buy_combine":
        return "You can only stack a pet onto the same pet below level 3."
    if name == "combine":
        return "Only matching pets below level 3 can be combined."
    return "That move is not allowed right now."
