"""
Battle Lab: recreate a battle from your Super Auto Pets screen and check the
engine against it. Runs locally with numpy only (no API keys, no internet).

How it reads the screen
-----------------------
The game's art is not part of this project, so the lab learns it from your
own screen:

* Layout: once, you click the 5 pet positions on each side of a battle frame
  (front first) and mark where a pet's attack number, health number, perk and
  level label sit relative to it. Everything is stored as fractions of the
  frame, so it works at any window size with the same aspect ratio.
* Memory: when you confirm what is on a frame (the lab pre-fills its guesses,
  you fix anything wrong), it keeps small pictures of each pet, each digit of
  the stats, each perk and each level label. Next time it compares what it
  sees with those pictures and picks the closest one.

Pictures are compared by their pattern (normalized correlation) and their
colours together, giving a match score from 0 to 1.
Numbers are split into digits by finding the bright characters, and each digit
is matched against digits learned from earlier frames. The more battles you
confirm, the better it gets; low scores are flagged for you to check.

Everything is kept in gui/lab_data/ (not in git): layout.json and memory.npz.
"""

import base64
import json
import os
import struct
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "lab_data")

SIDES = ("player", "opponent")
N_SLOTS = 5
PARTS = ("attack", "health", "perk", "level")
### Size of the pictures kept and compared (height, width)
PET_SIZE = (28, 28)
PERK_SIZE = (16, 16)
LEVEL_SIZE = (12, 24)
GLYPH_SIZE = (16, 11)
MAX_EXAMPLES = 24
### Below this score a guess is marked as unsure
SURE = {"pet": 0.75, "perk": 0.7, "level": 0.7, "digit": 0.6}


################################################################################
#### Images
################################################################################


def decode_rgb(b64, width, height):
    """A frame sent by the browser: raw RGB bytes, base64"""
    raw = base64.b64decode(b64)
    if len(raw) != width * height * 3:
        raise ValueError(
            f"frame is {len(raw)} bytes, expected {width}x{height}x3 = {width * height * 3}"
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def resize(img, out_h, out_w):
    """Area-average resize of an (h, w) or (h, w, c) array"""
    a = img.astype(np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, c = a.shape
    if h == 0 or w == 0:
        return np.zeros((out_h, out_w, c))
    ### Integral image, then the mean over each output cell's source area
    s = np.zeros((h + 1, w + 1, c))
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    ys = np.linspace(0, h, out_h + 1)
    xs = np.linspace(0, w, out_w + 1)

    def interp(table, y, x):
        ### Bilinear lookup in the integral image at fractional coordinates
        y0 = np.clip(np.floor(y).astype(int), 0, h - 1)
        x0 = np.clip(np.floor(x).astype(int), 0, w - 1)
        fy = (y - y0)[:, None, None]
        fx = (x - x0)[None, :, None]
        t00 = table[y0][:, x0]
        t01 = table[y0][:, x0 + 1]
        t10 = table[y0 + 1][:, x0]
        t11 = table[y0 + 1][:, x0 + 1]
        return (
            t00 * (1 - fy) * (1 - fx)
            + t01 * (1 - fy) * fx
            + t10 * fy * (1 - fx)
            + t11 * fy * fx
        )

    total = interp(s, ys, xs)
    cell = total[1:, 1:] - total[:-1, 1:] - total[1:, :-1] + total[:-1, :-1]
    area = np.outer(np.diff(ys), np.diff(xs))[:, :, None]
    out = cell / np.maximum(area, 1e-9)
    return out if img.ndim == 3 else out[:, :, 0]


def crop(img, box):
    """box: (center x, center y, width, height) as fractions of the frame"""
    h, w = img.shape[:2]
    cx, cy, bw, bh = box
    x0 = int(round((cx - bw / 2) * w))
    x1 = int(round((cx + bw / 2) * w))
    y0 = int(round((cy - bh / 2) * h))
    y1 = int(round((cy + bh / 2) * h))
    x0, x1 = max(0, x0), min(w, max(x1, x0 + 1))
    y0, y1 = max(0, y0), min(h, max(y1, y0 + 1))
    return img[y0:y1, x0:x1]


def feature(img, size):
    """A picture shrunk to a fixed size, as values from 0 to 1"""
    return (resize(img, *size) / 255.0).ravel()


def similarity(a, b):
    """
    How alike two pictures are, from 0 to 1: the average of their pattern
    correlation (ignores brightness) and how close their colours are. A flat
    patch (an empty perk spot) has no pattern, so only its colour counts.
    """
    close = 1.0 - float(np.abs(a - b).mean()) * 2.0
    da, db = a - a.mean(), b - b.mean()
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    if na < 1e-6 or nb < 1e-6:
        return max(close, 0.0)
    corr = float(da @ db) / (na * nb)
    return max((corr + close) / 2.0, 0.0)


def gray(img):
    a = img.astype(np.float64)
    return a if a.ndim == 2 else a @ np.array([0.299, 0.587, 0.114])


def png_b64(img):
    """A small PNG (grayscale or RGB uint8) as base64, for showing crops"""
    a = np.clip(np.asarray(img), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        color, rows = 0, a
    else:
        color, rows = 2, a.reshape(a.shape[0], -1)
    h, w = a.shape[:2]
    raw = b"".join(b"\x00" + bytes(r) for r in rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


################################################################################
#### Digits
################################################################################


def otsu(values):
    """Threshold that best splits values (0-255) into two groups"""
    hist, edges = np.histogram(values, bins=64, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 128
    centers = (edges[:-1] + edges[1:]) / 2
    best, best_t = -1, 128
    w0 = s0 = 0.0
    s_all = (hist * centers).sum()
    for i in range(len(hist) - 1):
        w0 += hist[i]
        s0 += hist[i] * centers[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        m0, m1 = s0 / w0, (s_all - s0) / w1
        between = w0 * w1 * (m0 - m1) ** 2
        if between > best:
            best, best_t = between, edges[i + 1]
    return best_t


def glyphs(img):
    """
    The characters in a number box, left to right, as binary pictures. The
    stats are light digits on a darker icon, so the bright pixels are kept;
    columns with no bright pixels separate the characters.
    """
    g = gray(img)
    if g.size == 0:
        return []
    ink = g > max(otsu(g.ravel()), 110)
    cols = ink.any(0)
    out = []
    x = 0
    w = ink.shape[1]
    min_w = max(1, w // 25)
    while x < w:
        if not cols[x]:
            x += 1
            continue
        start = x
        while x < w and cols[x]:
            x += 1
        piece = ink[:, start:x]
        rows = np.flatnonzero(piece.any(1))
        if len(rows) == 0 or (x - start) < min_w:
            continue
        out.append(piece[rows[0] : rows[-1] + 1].astype(np.float64) * 255)
    ### Specks much shorter than the tallest character are noise, not digits
    tallest = max((p.shape[0] for p in out), default=0)
    return [p for p in out if p.shape[0] >= tallest * 0.5]


################################################################################
#### Layout and memory
################################################################################


def layout_path():
    return os.path.join(DATA_DIR, "layout.json")


def memory_path():
    return os.path.join(DATA_DIR, "memory.npz")


def load_layout():
    try:
        with open(layout_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_layout(layout):
    """Problems with a layout, as a list of strings"""
    problems = []
    for side in SIDES:
        slots = layout.get(side)
        if not isinstance(slots, list) or len(slots) != N_SLOTS:
            problems.append(f"{side}: needs {N_SLOTS} pet positions")
    for key in ("slot",) + PARTS:
        box = layout.get(key)
        size = 2 if key == "slot" else 4
        if not isinstance(box, list) or len(box) != size:
            problems.append(f"{key}: missing")
    return problems


def save_layout(layout):
    problems = check_layout(layout)
    if problems:
        raise ValueError("; ".join(problems))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(layout_path(), "w") as f:
        json.dump(layout, f, indent=1)


class Memory:
    """Learned pictures: kind ("pet", "perk", "level", "digit") -> label -> vectors"""

    def __init__(self, items=None):
        self.items = items or {}

    @classmethod
    def load(cls, path=None):
        path = path or memory_path()
        items = {}
        if os.path.exists(path):
            with np.load(path) as z:
                for key in z.files:
                    kind, label, _ = key.split("|")
                    items.setdefault(kind, {}).setdefault(label, []).append(z[key])
        return cls(items)

    def save(self, path=None):
        path = path or memory_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = {
            f"{kind}|{label}|{i}": v
            for kind, labels in self.items.items()
            for label, vs in labels.items()
            for i, v in enumerate(vs)
        }
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    def add(self, kind, label, vec):
        examples = self.items.setdefault(kind, {}).setdefault(str(label), [])
        ### Skip near-duplicates of something already kept
        if any(similarity(v, vec) > 0.98 for v in examples):
            return False
        examples.append(vec)
        del examples[:-MAX_EXAMPLES]
        return True

    def match(self, kind, vec):
        """(best label, score, runner-up score); label None when nothing is known"""
        best, score, second = None, -1.0, -1.0
        for label, examples in self.items.get(kind, {}).items():
            s = max(similarity(v, vec) for v in examples)
            if s > score:
                best, score, second = label, s, score
            elif s > second:
                second = s
        return best, score, second

    def counts(self):
        return {kind: len(labels) for kind, labels in self.items.items()}


################################################################################
#### Reading a frame
################################################################################


def part_box(layout, side, slot, part):
    """Box of a pet slot or one of its parts, as frame fractions"""
    cx, cy = layout[side][slot]
    if part == "pet":
        w, h = layout["slot"]
        return (cx, cy, w, h)
    dx, dy, w, h = layout[part]
    ### Offsets are measured on your front pet and used for every pet. If the
    ###   game draws the opponent's stats mirrored, the layout says so
    if side == "opponent" and layout.get("mirror", False):
        dx = -dx
    return (cx + dx, cy + dy, w, h)


def read_number(img, memory):
    """(number or None, lowest digit score, glyph pictures)"""
    pieces = glyphs(img)
    if not pieces or "digit" not in memory.items:
        return None, 0.0, pieces
    digits, worst = "", 1.0
    for p in pieces:
        label, score, _ = memory.match("digit", feature(p, GLYPH_SIZE))
        digits += label
        worst = min(worst, score)
    try:
        return int(digits), worst, pieces
    except ValueError:
        return None, 0.0, pieces


def read_slot(img, layout, memory, side, slot):
    out = {"side": side, "slot": slot}
    pet_img = crop(img, part_box(layout, side, slot, "pet"))
    out["crop"] = png_b64(resize(pet_img, 48, 48))
    name, score, second = memory.match("pet", feature(pet_img, PET_SIZE))
    out["name"] = name
    out["name_score"] = round(float(score), 3)
    out["name_margin"] = round(float(score - second), 3)
    out["empty"] = name == "empty"
    for stat in ("attack", "health"):
        num_img = crop(img, part_box(layout, side, slot, stat))
        value, s, _ = read_number(num_img, memory)
        out[stat] = value
        out[stat + "_score"] = round(float(s), 3)
        out[stat + "_crop"] = png_b64(resize(num_img, 24, 32))
    for part, size in (("perk", PERK_SIZE), ("level", LEVEL_SIZE)):
        part_img = crop(img, part_box(layout, side, slot, part))
        label, s, _ = memory.match(part, feature(part_img, size))
        out[part] = label
        out[part + "_score"] = round(float(s), 3)
    ### Unsure when any guess is weak or missing (for an empty slot, only
    ###   whether it is empty matters)
    if out["empty"]:
        out["unsure"] = [] if score >= SURE["pet"] else ["name"]
        return out
    weak = [
        k
        for k, kind in (
            ("name", "pet"),
            ("perk", "perk"),
            ("level", "level"),
        )
        if out[k] is None or out[k + "_score"] < SURE[kind]
    ]
    weak += [
        s
        for s in ("attack", "health")
        if out[s] is None or out[s + "_score"] < SURE["digit"]
    ]
    out["unsure"] = weak
    return out


def read_frame(img, layout=None, memory=None):
    """
    What the lab sees in a frame: for each side, the 5 slots front first,
    each with its guesses and scores. Returns {"player": [...], "opponent": [...]}.
    """
    layout = layout or load_layout()
    if layout is None:
        raise ValueError("Set up the screen layout first")
    memory = memory or Memory.load()
    return {
        side: [read_slot(img, layout, memory, side, i) for i in range(N_SLOTS)]
        for side in SIDES
    }


def teach(img, labels, layout=None, memory=None):
    """
    Learn from a frame whose contents you confirmed. labels: {"player": [...],
    "opponent": [...]}, 5 entries per side front first, each None (leave this
    slot out), {"empty": true}, or {"name", "attack", "health", "level", "perk"}.
    Returns how many new pictures were kept, by kind.
    """
    layout = layout or load_layout()
    if layout is None:
        raise ValueError("Set up the screen layout first")
    memory = memory or Memory.load()
    added = {"pet": 0, "digit": 0, "perk": 0, "level": 0}
    for side in SIDES:
        for i, label in enumerate((labels.get(side) or [])[:N_SLOTS]):
            if not label:
                continue
            pet_img = crop(img, part_box(layout, side, i, "pet"))
            name = "empty" if label.get("empty") else label["name"]
            added["pet"] += memory.add("pet", name, feature(pet_img, PET_SIZE))
            if label.get("empty"):
                continue
            for stat in ("attack", "health"):
                value = label.get(stat)
                if value is None:
                    continue
                pieces = glyphs(crop(img, part_box(layout, side, i, stat)))
                text = str(int(value))
                ### Only when the characters found match the number typed
                if len(pieces) == len(text):
                    for ch, p in zip(text, pieces):
                        added["digit"] += memory.add(
                            "digit", ch, feature(p, GLYPH_SIZE)
                        )
            for part, size in (("perk", PERK_SIZE), ("level", LEVEL_SIZE)):
                value = label.get(part)
                if value is None:
                    continue
                part_img = crop(img, part_box(layout, side, i, part))
                added[part] += memory.add(part, str(value), feature(part_img, size))
    memory.save()
    return added


################################################################################
#### Recreating and checking the battle
################################################################################


def spec(pet):
    """A pet as read or confirmed, as a sapai.scenarios pet description"""
    perk = pet.get("perk") or "none"
    d = {
        "name": pet["name"],
        "attack": int(pet["attack"]),
        "health": int(pet["health"]),
        "level": int(pet.get("level") or 1),
    }
    if perk != "none":
        d["perk"] = perk
    return d


def pets_of(side):
    """The pets of one side, front first, without empty slots"""
    return [p for p in side if p and not p.get("empty") and p.get("name")]


def one_line(pet):
    s = spec(pet)
    text = f"{s['name']} {s['attack']}/{s['health']}"
    if s["level"] > 1:
        text += f" L{s['level']}"
    if s.get("perk"):
        text += f" {s['perk']}"
    return text


def outcome_of(player_left, opponent_left):
    if player_left and not opponent_left:
        return "win"
    if opponent_left and not player_left:
        return "loss"
    if not player_left and not opponent_left:
        return "draw"
    return None


def run(start, end=None, runs=200):
    """
    Recreate the battle from the start-of-battle teams and check it.
    start/end: {"player": [pets], "opponent": [pets]} (end is optional: the
    survivors the game showed). Returns the replay frames of one battle, how
    often the engine gives each result over `runs` battles, and, with an end,
    whether the game's ending is one the engine can produce.
    """
    from gui import game
    from sapai.scenarios import describe, make_team, run_battle

    team = [spec(p) for p in pets_of(start["player"])]
    enemy = [spec(p) for p in pets_of(start["opponent"])]
    if not team and not enemy:
        raise ValueError("Both teams are empty")
    scenario = {"team": team, "enemy": enemy}

    np.random.seed(0)
    frames, result = game.replay_battle(
        make_team(team), make_team(enemy), names=("You", "Opponent")
    )
    results = {"win": 0, "loss": 0, "draw": 0}
    endings = {}
    for seed in range(runs):
        r, t0, t1 = run_battle(scenario, seed)
        results[r] += 1
        key = (r, tuple(t0), tuple(t1))
        endings[key] = endings.get(key, 0) + 1

    out = {
        "frames": frames,
        "outcome": {0: "win", 1: "loss", 2: "draw"}[result],
        "odds": {k: v / runs for k, v in results.items()},
        "endings": [
            {
                "result": k[0],
                "player": list(k[1]),
                "opponent": list(k[2]),
                "share": v / runs,
            }
            for k, v in sorted(endings.items(), key=lambda kv: -kv[1])[:8]
        ],
        "team": [one_line(p) for p in pets_of(start["player"])],
        "enemy": [one_line(p) for p in pets_of(start["opponent"])],
    }
    if end is not None:
        game_player = [one_line(p) for p in pets_of(end.get("player", []))]
        game_opponent = [one_line(p) for p in pets_of(end.get("opponent", []))]
        game_result = end.get("result") or outcome_of(game_player, game_opponent)
        ### Compare the way the scenarios do: names and stats as shown
        want_p = describe(make_team([spec(p) for p in pets_of(end.get("player", []))]))
        want_o = describe(
            make_team([spec(p) for p in pets_of(end.get("opponent", []))])
        )
        same_result = results.get(game_result, 0) / runs if game_result else None
        same_ending = endings.get((game_result, tuple(want_p), tuple(want_o)), 0) / runs
        out["game"] = {
            "result": game_result,
            "player": game_player,
            "opponent": game_opponent,
        }
        out["match"] = {
            "result_share": same_result,
            "ending_share": same_ending,
            "matches": same_ending > 0,
        }
    return out


SCENARIO_DIR = os.path.join(HERE, "..", "tests", "scenarios")


def save(start, end, note="", runs=200):
    """
    Save a captured battle as a scenario. One whose ending the engine can
    produce goes into tests/scenarios/from_game.yaml (verified); one it can't
    into tests/scenarios/game_discrepancies.yaml, to be looked at and fixed.
    Returns the file it was written to and the check.
    """
    checked = run(start, end, runs)
    if "match" not in checked:
        raise ValueError("Add the end of the battle before saving")
    ok = checked["match"]["matches"]
    name = "from_game.yaml" if ok else "game_discrepancies.yaml"
    path = os.path.join(SCENARIO_DIR, name)
    lines = [
        "",
        f"- name: {json.dumps(_title(checked))}",
        "  source: real game, captured with the Battle Lab",
        "  verified: true",
        "  seen: true",
    ]
    if not ok:
        lines.append("  discrepancy: true")
    if note:
        lines.append(f"  note: {json.dumps(note)}")
    lines.append(f"  team: {json.dumps(checked['team'])}")
    lines.append(f"  enemy: {json.dumps(checked['enemy'])}")
    lines.append(f"  result: {checked['game']['result']}")
    lines.append(
        "  survivors: {team: "
        + json.dumps(checked["game"]["player"])
        + ", enemy: "
        + json.dumps(checked["game"]["opponent"])
        + "}"
    )
    if not ok:
        odds = ", ".join(f"{k} {v:.0%}" for k, v in checked["odds"].items())
        lines.append(f"  # The engine never ends this way; its results: {odds}")
    new_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if new_file:
            f.write(
                "# Battles captured from the real game with the Battle Lab (gui/lab.py).\n"
                if ok
                else "# Battles from the real game that the engine never reproduces.\n"
                "# Each one is a rule to look into; the test suite expects them to fail\n"
                "# until the engine is fixed.\n"
            )
        f.write("\n".join(lines) + "\n")
    return {
        "file": os.path.relpath(path, os.path.join(HERE, "..")),
        "check": checked["match"],
    }


def _title(checked):
    t = ", ".join(checked["team"]) or "nothing"
    e = ", ".join(checked["enemy"]) or "nothing"
    return f"{t} vs {e}"
