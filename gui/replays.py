"""
Replay catalog for sapai arena.

Every match played through the GUI (Play and Watch mode, not series) is kept
in ``gui/replays/``: ``<id>.json.gz`` holds the whole game (every round's
teams, shop actions, the models' reasoning, each battle step, the game log and
the chat) and ``index.json`` holds one summary line per game for the list. The
file is rewritten after every round, so unfinished games are kept too.

Games saved before the catalog existed are imported from ``gui/saves/`` with
their game log, so their battles can be read but not stepped through.
"""

import datetime
import glob
import gzip
import json
import os

from gui.game import _now

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "replays")
INDEX = os.path.join(DIR, "index.json")


def _valid(rid):
    return (
        isinstance(rid, str)
        and len(rid) == 32
        and all(c in "0123456789abcdef" for c in rid)
    )


def _path(rid):
    return os.path.join(DIR, rid + ".json.gz")


def _load_index():
    try:
        with open(INDEX) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_index(idx):
    os.makedirs(DIR, exist_ok=True)
    tmp = INDEX + ".tmp"
    with open(tmp, "w") as f:
        json.dump(idx, f)
    os.replace(tmp, INDEX)


def _write(rid, doc):
    os.makedirs(DIR, exist_ok=True)
    tmp = _path(rid) + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(doc, f, default=str)
    os.replace(tmp, _path(rid))


def summary(rid, match, previous=None):
    """The catalog line for a match: who played, how it went."""
    previous = previous or {}
    a, b = match.seats
    labels = [match.label(0), match.label(1)]
    human = a.kind == "human"
    if match.over:
        winner = labels[0] if match.result == "seat0" else labels[1]
        status = "finished"
        result = (
            ("You won" if match.result == "seat0" else f"{labels[1]} won")
            if human
            else f"{winner} won"
        )
    else:
        status, result, winner = "unfinished", "Unfinished", None
    hist = match.history
    return {
        "id": rid,
        "title": previous.get("title")
        or (f"You vs {labels[1]}" if human else f"{labels[0]} vs {labels[1]}"),
        "note": previous.get("note", ""),
        "mode": "play" if human else "watch",
        "players": labels,
        "kinds": [a.kind, b.kind],
        "created": match.created,
        "updated": _now(),
        "status": status,
        "result": result,
        "winner": winner,
        "turns": len(hist),
        "history": hist,
        "score": [
            {"wins": s.wins, "lives": max(0, int(s.player.lives))} for s in match.seats
        ],
        "battles": {
            "win": hist.count("win"),
            "loss": hist.count("loss"),
            "draw": hist.count("draw"),
        },
        "steps": True,
    }


def record(rid, match):
    """Write (or update) the replay for a match. Called after every round."""
    if not _valid(rid):
        return
    idx = _load_index()
    meta = summary(rid, match, idx.get(rid))
    _write(
        rid,
        {
            "meta": meta,
            "rounds": match.rounds,
            "events": match.events,
            "chat": match.chat,
        },
    )
    idx[rid] = meta
    _save_index(idx)


def load_rounds(rid):
    """Rounds already recorded for a match, so a resumed match keeps them."""
    if not _valid(rid) or not os.path.exists(_path(rid)):
        return []
    try:
        with gzip.open(_path(rid), "rt", encoding="utf-8") as f:
            return json.load(f).get("rounds", [])
    except (OSError, ValueError):
        return []


def listing():
    idx = _load_index()
    return sorted(idx.values(), key=lambda m: m.get("updated", ""), reverse=True)


def get(rid):
    if not _valid(rid) or not os.path.exists(_path(rid)):
        raise Exception("That replay doesn't exist.")
    with gzip.open(_path(rid), "rt", encoding="utf-8") as f:
        doc = json.load(f)
    doc["meta"] = _load_index().get(rid, doc.get("meta"))
    return doc


def rename(rid, title=None, note=None):
    idx = _load_index()
    if rid not in idx:
        raise Exception("That replay doesn't exist.")
    if title is not None:
        idx[rid]["title"] = str(title).strip()[:120] or idx[rid]["title"]
    if note is not None:
        idx[rid]["note"] = str(note).strip()[:1000]
    _save_index(idx)
    return idx[rid]


def delete(rid):
    idx = _load_index()
    if rid in idx:
        del idx[rid]
        _save_index(idx)
    if _valid(rid) and os.path.exists(_path(rid)):
        os.remove(_path(rid))


def import_saves(saves_dir, from_save):
    """Catalog games saved before replays existed, from their game log."""
    idx = _load_index()
    added = 0
    for path in glob.glob(os.path.join(saves_dir, "*.json")):
        rid = os.path.basename(path)[:-5]
        if not _valid(rid) or rid in idx:
            continue
        try:
            with open(path) as f:
                d = json.load(f)
            m = from_save(d)
        except Exception:
            continue
        if not m.history:
            continue
        rounds = []
        battles = [e for e in m.events if e.get("kind") == "battle"]
        for k, e in enumerate(battles):
            rounds.append(
                {
                    "turn": k + 1,
                    "outcome": m.history[k] if k < len(m.history) else "draw",
                    "lines": e.get("lines", []),
                    "frames": [],
                    "shopLogs": [[], []],
                    "thoughts": [[], []],
                }
            )
        m.rounds = rounds
        m.created = d.get("created") or _now()
        meta = summary(rid, m)
        meta["steps"] = False  # battles from the log only, no step-by-step frames
        meta["created"] = meta["updated"] = _mtime(path)
        _write(
            rid, {"meta": meta, "rounds": rounds, "events": m.events, "chat": m.chat}
        )
        idx[rid] = meta
        added += 1
    if added:
        _save_index(idx)
    return added


def _mtime(path):
    return (
        datetime.datetime.fromtimestamp(os.path.getmtime(path))
        .astimezone()
        .isoformat(timespec="seconds")
    )
