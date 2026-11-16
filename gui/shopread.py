"""
Shop Reader: reads the shop screen of the real game and suggests moves.

It works like the Battle Lab (gui/lab.py) and shares what that has learned
(digits, levels, perks, your team's pets), but has its own screen layout,
because the shop screen is laid out differently:

    team       your 5 team slots, front pet first
    shop       every shop slot, left to right (pets and food alike; which
               one a slot holds is read, not assumed)
    slot       size of a team pet's picture box
    shop_slot  size of a shop slot's picture box
    attack, health, perk, level     boxes around your front pet's parts,
               as offsets from its slot (used for every team slot)
    shop_attack, shop_health        the same for the first shop slot
    gold, lives, turn, wins         boxes around those numbers (whole screen)

Positions and boxes are fractions of the captured screen. Pictures it
learns here: what a shop slot holds ("item": a pet, a food or "empty") and
whether it is frozen ("frozen": "yes" or "no").

Then the screen is turned into the engine's Player, and a trained model
says what it would do: suggest() gives the top moves, each with the
model's probability for it, in plain words with the slot numbers you see.
"""

import glob
import json
import os

import numpy as np

from gui import lab
from gui.lab import Memory, crop, feature, glyphs, png_b64, read_number, resize
from gui.lab import GLYPH_SIZE, LEVEL_SIZE, PERK_SIZE, PET_SIZE, SURE

N_TEAM = 5
TEAM_PARTS = ("attack", "health", "perk", "level")
SHOP_PARTS = ("shop_attack", "shop_health")
COUNTERS = ("gold", "lives", "turn", "wins")
SURE_ITEM = 0.75
### Frozen slots all look alike (ice), unfrozen ones as different as what is
###   in them: a slot this close to a known frozen one is frozen, one below
###   the lower mark is not, and in between it is a guess
FROZEN_YES = 0.85
FROZEN_NO = 0.75
### Captures are kept at most this wide (the reader works on fractions)
CAPTURE_MAX_W = 1600


################################################################################
#### Names
################################################################################


def _names():
    from sapai.data import data

    pets = {
        p["name"]: k
        for k, p in data["pets"].items()
        if k != "pet-none" and p.get("name") not in (None, "none")
    }
    foods = {
        f["name"]: k
        for k, f in data["foods"].items()
        if k != "food-none" and f.get("name") not in (None, "none")
    }
    perks = {s["name"]: k for k, s in data["statuses"].items()}
    return pets, foods, perks


PET_IDS, FOOD_IDS, PERK_IDS = _names()


def canonical(name, table):
    """The name as the game data spells it, or None"""
    if not name:
        return None
    low = str(name).strip().lower()
    for n in table:
        if n.lower() == low:
            return n
    return None


################################################################################
#### Pictures of shop slots
################################################################################


def item_feature(pic):
    """
    A shop slot's picture with any tint taken out (each colour channel
    scaled to the same spread): ice over a frozen slot tints and fades it,
    and the pet or food under it should still be known
    """
    a = resize(pic, *PET_SIZE).reshape(-1, 3).astype(np.float64)
    a = (a - a.mean(0)) / (a.std(0) + 8.0)
    return np.clip(a * 0.15 + 0.5, 0, 1).ravel()


def is_frozen(memory, pic):
    """(frozen, sure)"""
    vec = frozen_feature(pic)
    known = memory.items.get("frozen", {})

    def best(label):
        examples = known.get(label) or []
        return max((lab.similarity(v, vec) for v in examples), default=None)

    yes, no = best("yes"), best("no")
    if yes is None:
        ### No frozen slot seen yet: nothing to tell it by
        return False, no is not None and no >= FROZEN_YES
    if yes >= FROZEN_YES:
        return True, True
    if yes < FROZEN_NO:
        return False, True
    return (no is None or yes > no), False


def frozen_feature(pic):
    """
    What ice does to a slot, whatever is in it: its colour (a blue cast)
    and how washed out it is (less contrast)
    """
    a = pic.reshape(-1, 3).astype(np.float64) / 255.0
    return np.concatenate([a.mean(0), a.std(0) * 2])


################################################################################
#### Layout
################################################################################


def layout_path():
    return os.path.join(lab.DATA_DIR, "shop_layout.json")


def capture_path():
    return os.path.join(lab.DATA_DIR, "shop_capture.npy")


def load_layout():
    try:
        with open(layout_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_layout(layout):
    problems = []
    team = layout.get("team")
    if not isinstance(team, list) or len(team) != N_TEAM:
        problems.append(f"team: needs {N_TEAM} pet positions")
    shop = layout.get("shop")
    if not isinstance(shop, list) or not shop:
        problems.append("shop: needs the shop slot positions")
    for key in ("slot", "shop_slot"):
        if not isinstance(layout.get(key), list) or len(layout[key]) != 2:
            problems.append(f"{key}: missing")
    for key in TEAM_PARTS + SHOP_PARTS + COUNTERS:
        if not isinstance(layout.get(key), list) or len(layout[key]) != 4:
            problems.append(f"{key}: missing")
    return problems


def save_layout(layout):
    problems = check_layout(layout)
    if problems:
        raise ValueError("; ".join(problems))
    os.makedirs(lab.DATA_DIR, exist_ok=True)
    with open(layout_path(), "w") as f:
        json.dump(layout, f, indent=1)


def box(layout, group, i, part):
    """Box (center x, center y, width, height) of a slot or one of its parts"""
    cx, cy = layout[group][i]
    if part == "pet":
        w, h = layout["slot" if group == "team" else "shop_slot"]
        return (cx, cy, w, h)
    dx, dy, w, h = layout[part]
    return (cx + dx, cy + dy, w, h)


################################################################################
#### Captures (the overlay saves what it saw, the set-up page shows it)
################################################################################


def shrink(img, max_w=CAPTURE_MAX_W):
    h, w = img.shape[:2]
    if w <= max_w:
        return np.ascontiguousarray(img[:, :, :3]).astype(np.uint8)
    nh = int(round(h * max_w / w))
    return np.clip(resize(img[:, :, :3], nh, max_w), 0, 255).astype(np.uint8)


def save_capture(img):
    os.makedirs(lab.DATA_DIR, exist_ok=True)
    tmp = capture_path() + ".tmp.npy"
    np.save(tmp, shrink(img))
    os.replace(tmp, capture_path())


def load_capture():
    try:
        return np.load(capture_path())
    except OSError:
        return None


################################################################################
#### Reading
################################################################################


def _number(img, memory, b):
    part = crop(img, b)
    value, score, _ = read_number(part, memory)
    return value, float(score), part


def read_team_slot(img, layout, memory, i):
    out = {"slot": i}
    pic = crop(img, box(layout, "team", i, "pet"))
    out["crop"] = png_b64(resize(pic, 48, 48))
    name, score, _ = memory.match("pet", feature(pic, PET_SIZE))
    out["name"] = name
    out["name_score"] = round(float(score), 3)
    out["empty"] = name == "empty"
    if out["empty"]:
        out["unsure"] = [] if score >= SURE["pet"] else ["name"]
        return out
    unsure = [] if name and score >= SURE["pet"] else ["name"]
    for stat in ("attack", "health"):
        value, s, _ = _number(img, memory, box(layout, "team", i, stat))
        out[stat] = value
        if value is None or s < SURE["digit"]:
            unsure.append(stat)
    for part, size in (("perk", PERK_SIZE), ("level", LEVEL_SIZE)):
        label, s, _ = memory.match(
            part, feature(crop(img, box(layout, "team", i, part)), size)
        )
        out[part] = label
        if label is None or s < SURE[part]:
            unsure.append(part)
    out["unsure"] = unsure
    return out


def read_shop_slot(img, layout, memory, i):
    out = {"slot": i}
    pic = crop(img, box(layout, "shop", i, "pet"))
    out["crop"] = png_b64(resize(pic, 48, 48))
    name, score, _ = memory.match("item", item_feature(pic))
    out["name"] = name
    out["name_score"] = round(float(score), 3)
    if name is None or name == "empty":
        out["kind"] = "empty" if name == "empty" else None
    elif name in PET_IDS:
        out["kind"] = "pet"
    elif name in FOOD_IDS:
        out["kind"] = "food"
    else:
        out["kind"] = None
    unsure = [] if out["kind"] and score >= SURE_ITEM else ["name"]
    if out["kind"] == "empty":
        out["unsure"] = unsure
        return out
    out["frozen"], sure = is_frozen(memory, pic)
    if not sure:
        unsure.append("frozen")
    if out["kind"] == "pet":
        for stat in ("attack", "health"):
            value, s, _ = _number(img, memory, box(layout, "shop", i, "shop_" + stat))
            out[stat] = value
            if value is None or s < SURE["digit"]:
                unsure.append(stat)
    out["unsure"] = unsure
    return out


def read_shop(img, layout=None, memory=None):
    """
    What the shop screen shows: {"team": [5 slots front first], "shop":
    [slots left to right], "counters": {"gold", "lives", "turn", "wins"},
    "unsure": [what to check]}
    """
    layout = layout or load_layout()
    if layout is None:
        raise ValueError("Set up the shop screen layout first")
    memory = memory or Memory.load()
    out = {
        "team": [read_team_slot(img, layout, memory, i) for i in range(N_TEAM)],
        "shop": [
            read_shop_slot(img, layout, memory, i) for i in range(len(layout["shop"]))
        ],
        "counters": {},
    }
    unsure = []
    for key in COUNTERS:
        value, s, _ = _number(img, memory, tuple(layout[key]))
        out["counters"][key] = value
        if value is None or s < SURE["digit"]:
            unsure.append(key)
    for i, t in enumerate(out["team"]):
        unsure += [f"team {i + 1} {k}" for k in t["unsure"]]
    for i, s in enumerate(out["shop"]):
        unsure += [f"shop {i + 1} {k}" for k in s["unsure"]]
    out["unsure"] = unsure
    return out


################################################################################
#### Teaching
################################################################################


def _teach_number(memory, part_img, value):
    if value is None or value == "":
        return 0
    pieces = glyphs(part_img)
    text = str(int(value))
    ### Only when the characters found match the number typed
    if len(pieces) != len(text):
        return 0
    return sum(
        memory.add("digit", ch, feature(p, GLYPH_SIZE)) for ch, p in zip(text, pieces)
    )


def teach_shop(img, labels, layout=None, memory=None):
    """
    Learn from a shop screen whose contents you confirmed. labels:
      "team": 5 entries front first: None (skip), {"empty": true} or
              {"name", "attack", "health", "level", "perk"}
      "shop": one entry per shop slot: None, {"empty": true}, a pet
              {"name", "attack", "health", "frozen"} or a food {"name", "frozen"}
      "counters": {"gold", "lives", "turn", "wins"} (any of them)
    Returns how many new pictures were kept, by kind.
    """
    layout = layout or load_layout()
    if layout is None:
        raise ValueError("Set up the shop screen layout first")
    memory = memory or Memory.load()
    added = {"pet": 0, "item": 0, "frozen": 0, "digit": 0, "perk": 0, "level": 0}
    for i, label in enumerate((labels.get("team") or [])[:N_TEAM]):
        if not label:
            continue
        pic = crop(img, box(layout, "team", i, "pet"))
        if label.get("empty"):
            added["pet"] += memory.add("pet", "empty", feature(pic, PET_SIZE))
            continue
        name = canonical(label.get("name"), PET_IDS)
        if not name:
            raise ValueError(f"Team slot {i + 1}: '{label.get('name')}' isn't a pet")
        added["pet"] += memory.add("pet", name, feature(pic, PET_SIZE))
        for stat in ("attack", "health"):
            part = crop(img, box(layout, "team", i, stat))
            added["digit"] += _teach_number(memory, part, label.get(stat))
        for part, size in (("perk", PERK_SIZE), ("level", LEVEL_SIZE)):
            value = label.get(part)
            if value is not None and value != "":
                pic2 = crop(img, box(layout, "team", i, part))
                added[part] += memory.add(part, str(value), feature(pic2, size))
    for i, label in enumerate((labels.get("shop") or [])[: len(layout["shop"])]):
        if not label:
            continue
        pic = crop(img, box(layout, "shop", i, "pet"))
        if label.get("empty"):
            added["item"] += memory.add("item", "empty", item_feature(pic))
            continue
        name = canonical(label.get("name"), PET_IDS) or canonical(
            label.get("name"), FOOD_IDS
        )
        if not name:
            raise ValueError(
                f"Shop slot {i + 1}: '{label.get('name')}' isn't a pet or food"
            )
        added["item"] += memory.add("item", name, item_feature(pic))
        added["frozen"] += memory.add(
            "frozen", "yes" if label.get("frozen") else "no", frozen_feature(pic)
        )
        if name in PET_IDS:
            for stat in ("attack", "health"):
                part = crop(img, box(layout, "shop", i, "shop_" + stat))
                added["digit"] += _teach_number(memory, part, label.get(stat))
    for key, value in (labels.get("counters") or {}).items():
        if key in COUNTERS:
            part = crop(img, tuple(layout[key]))
            added["digit"] += _teach_number(memory, part, value)
    memory.save()
    return added


################################################################################
#### The engine's view of the screen
################################################################################


def to_player(state):
    """
    A sapai Player for what the screen shows. state is read_shop()'s output
    (after any corrections): team, shop and counters. Returns (player,
    screen_slots): screen_slots[k] is the screen's shop slot number (from 1)
    of the engine's shop slot k, so moves can be told in screen terms.
    """
    from sapai.foods import Food
    from sapai.pets import Pet
    from sapai.player import Player
    from sapai.shop import Shop, ShopSlot
    from sapai.teams import Team

    counters = state.get("counters") or {}
    turn = int(counters.get("turn") or 1)
    pets = []
    for i, t in enumerate(state.get("team") or []):
        if not t or t.get("empty"):
            continue
        name = canonical(t.get("name"), PET_IDS)
        if not name:
            raise ValueError(f"Team slot {i + 1}: unknown pet '{t.get('name')}'")
        pet = Pet(PET_IDS[name])
        pet.level = int(t.get("level") or 1)
        pet._attack = int(t.get("attack") or 1)
        pet._health = int(t.get("health") or 1)
        perk = canonical(t.get("perk"), PERK_IDS)
        pet.status = PERK_IDS[perk] if perk else "none"
        pets.append(pet)
    slots, screen_slots = [], []
    for i, s in enumerate(state.get("shop") or []):
        if not s or s.get("empty") or s.get("kind") == "empty" or not s.get("name"):
            continue
        pet_name = canonical(s.get("name"), PET_IDS)
        food_name = canonical(s.get("name"), FOOD_IDS)
        if pet_name:
            pet = Pet(PET_IDS[pet_name])
            if s.get("attack") is not None:
                pet._attack = int(s["attack"])
            if s.get("health") is not None:
                pet._health = int(s["health"])
            slot = ShopSlot(pet)
        elif food_name:
            slot = ShopSlot(Food(FOOD_IDS[food_name]))
        else:
            raise ValueError(
                f"Shop slot {i + 1}: unknown pet or food '{s.get('name')}'"
            )
        slot.frozen = bool(s.get("frozen"))
        slots.append(slot)
        screen_slots.append(i + 1)
    ### Pets first, then food, as the engine's shop keeps them
    order = sorted(range(len(slots)), key=lambda k: slots[k].slot_type == "food")
    slots = [slots[k] for k in order]
    screen_slots = [screen_slots[k] for k in order]
    shop = Shop(slots, turn=turn) if slots else Shop([], turn=turn, fixed_rules=True)
    player = Player(team=Team(pets), shop=shop)
    player.turn = turn
    player.gold = int(counters.get("gold") or 0)
    player.lives = int(counters.get("lives") or 1)
    player.wins = int(counters.get("wins") or 0)
    return player, screen_slots


################################################################################
#### Moves
################################################################################


def move_text(player, action, screen_slots=None):
    """A shop action as an instruction, with the slot numbers on screen"""
    from sapai.rl import env as E

    name, args = E.ACTIONS[action]
    team = E.team_pets(player)
    shop = player.shop.slots
    screen_slots = screen_slots or list(range(1, len(shop) + 1))

    def item(i):
        from sapai.data import data

        obj = shop[i].obj
        table = data["foods"] if shop[i].slot_type == "food" else data["pets"]
        return f"{table[obj.name]['name']} (shop slot {screen_slots[i]})"

    def pet(j):
        from sapai.data import data

        where = "front pet" if j == 0 else f"team slot {j + 1}"
        return f"{data['pets'][team[j].name]['name']} ({where})"

    texts = {
        "end_turn": lambda: "End turn",
        "roll": lambda: "Roll",
        "buy_pet": lambda: f"Buy {item(args[0])}",
        "buy_combine": lambda: f"Buy {item(args[0])} and merge it onto {pet(args[1])}",
        "buy_food": lambda: f"Feed {item(args[0])} to {pet(args[1])}",
        "buy_food_untargeted": lambda: f"Buy {item(args[0])}",
        "sell": lambda: f"Sell {pet(args[0])}",
        "freeze": lambda: ("Unfreeze " if shop[args[0]].frozen else "Freeze ")
        + item(args[0]),
        "swap": lambda: f"Swap {pet(args[0])} and {pet(args[1])}",
        "combine": lambda: f"Merge {pet(args[1])} onto {pet(args[0])}",
    }
    try:
        return texts[name]()
    except Exception:
        return name.replace("_", " ")


def suggest(policy, player, screen_slots=None, top=3):
    """
    The model's top moves for this position: {"moves": [{"text", "prob",
    "action"}], "legal": number of legal moves, "value": the model's
    estimate of how well the game is going}. prob is the model's
    probability for the move among the legal ones.
    """
    from sapai.rl import env as E

    obs = E.encode(player, 0, None, 25, int(getattr(player, "wins", 0) or 0))
    mask = E.action_mask(player)
    _, z = policy.predict(obs, mask)
    legal = np.flatnonzero(mask)
    zz = np.asarray(z, dtype=np.float64)[legal]
    pr = np.exp(zz - zz.max())
    pr /= pr.sum()
    order = np.argsort(-pr)[:top]
    moves = [
        {
            "action": int(legal[k]),
            "text": move_text(player, int(legal[k]), screen_slots),
            "prob": round(float(pr[k]), 4),
        }
        for k in order
    ]
    try:
        value = float(policy.value(obs))
    except Exception:
        value = None
    return {"moves": moves, "legal": int(len(legal)), "value": value}


################################################################################
#### Models
################################################################################

MODELS_DIR = os.path.join(lab.HERE, "models")


def list_models():
    """[(name, label)] of the models in gui/models, best rated first"""
    from gui.game import MODEL_INFO

    out = []
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, "*.npz"))):
        name = os.path.basename(path)[:-4]
        label, rating = MODEL_INFO.get(name, (name, None))
        out.append((name, f"{label} (rated {rating})" if rating else label, rating))
    out.sort(key=lambda m: -(m[2] or 0))
    return [(n, label) for n, label, _ in out]


_POLICIES = {}


def load_policy(name):
    if name not in _POLICIES:
        from gui.game import LAYOUT
        from gui.npmodel import NumpyPolicy

        with np.load(os.path.join(MODELS_DIR, name + ".npz")) as f:
            _POLICIES[name] = NumpyPolicy(dict(f), layout=LAYOUT)
    return _POLICIES[name]


def advise(img, model, top=3, layout=None, memory=None):
    """Read a shop screen and suggest moves: the overlay's one call"""
    seen = read_shop(img, layout, memory)
    player, screen_slots = to_player(seen)
    out = suggest(load_policy(model), player, screen_slots, top)
    out["seen"] = seen
    return out
