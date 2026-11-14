"""
Game log for the GUI: records what every action changed so the rules can be
checked by eye.

Each shop action is logged with a snapshot of the player before and after:
gold, team pets, shop items. The difference is written out in plain words
("shop Otter 1/4 -> 1/5"), together with what the basic rules say the gold
change should be and the ability text of the pets involved, so any extra change
can be traced to the ability that caused it. Hard rule breaks (more than 5 pets,
negative gold, level above 3) are flagged.
"""

import random

from sapai.data import data


def _short(name):
    name = str(name)
    return (name.split("-", 1)[1] if "-" in name else name).replace("-", " ").title()


def _perk(status):
    if status in (None, "none"):
        return None
    return data["statuses"].get(status, {}).get("name", _short(status))


def uid(obj):
    """Stable id for a pet or food object. id() is reused after an object is
    freed (every roll), which would make new shop items look like old ones.
    Random so ids stay unique across saved games and server restarts."""
    u = getattr(obj, "_gui_uid", None)
    if u is None:
        u = random.getrandbits(62)
        try:
            obj._gui_uid = u
        except Exception:
            return id(obj)
    return u


def snapshot(p):
    """Everything the log compares, keyed by object identity."""
    team = []
    for slot in p.team:
        if slot.empty:
            continue
        x = slot.pet
        team.append({"id": uid(x), "name": _short(x.name), "pet": x.name,
                     "atk": int(x.attack), "hp": int(x.health), "lvl": int(x.level),
                     "xp": int(x.experience), "perk": _perk(x.status),
                     "sell": int(getattr(x, "sell_value", x.level))})
    shop = []
    for i, s in enumerate(p.shop.slots):
        if s.empty:
            continue
        o = s.obj
        item = {"id": uid(o), "i": i, "name": _short(o.name), "kind": s.slot_type,
                "cost": int(s.cost), "frozen": bool(s.frozen)}
        if s.slot_type != "food":
            item.update(atk=int(o.attack), hp=int(o.health))
        shop.append(item)
    return {"gold": int(p.gold), "lives": int(p.lives), "team": team, "shop": shop}


def _pet_str(x):
    s = f"{x['name']} {x['atk']}/{x['hp']}"
    if x.get("lvl", 1) > 1:
        s += f" L{x['lvl']}"
    if x.get("perk"):
        s += f" with {x['perk']}"
    return s


def _item_str(x):
    if x["kind"] == "food":
        return x["name"]
    return f"{x['name']} {x['atk']}/{x['hp']}"


def diff(a, b, roll=False):
    """Plain-language list of what changed between two snapshots."""
    lines = []
    if a["gold"] != b["gold"]:
        d = b["gold"] - a["gold"]
        lines.append(f"gold {a['gold']} → {b['gold']} ({'+' if d > 0 else '−'}{abs(d)})")

    ta = {x["id"]: x for x in a["team"]}
    tb = {x["id"]: x for x in b["team"]}
    sa = {x["id"]: x for x in a["shop"]}
    for x in b["team"]:
        if x["id"] not in ta:
            src = " from the shop" if x["id"] in sa else ""
            lines.append(f"team: {_pet_str(x)} joined{src}")
    for x in a["team"]:
        if x["id"] not in tb:
            lines.append(f"team: {_pet_str(x)} left the team")
    for x in b["team"]:
        old = ta.get(x["id"])
        if not old:
            continue
        ch = []
        if (old["atk"], old["hp"]) != (x["atk"], x["hp"]):
            ch.append(f"{old['atk']}/{old['hp']} → {x['atk']}/{x['hp']}")
        if old["lvl"] != x["lvl"]:
            ch.append(f"level {old['lvl']} → {x['lvl']}")
        elif old["xp"] != x["xp"]:
            ch.append(f"experience {old['xp']} → {x['xp']}")
        if old["perk"] != x["perk"]:
            ch.append(f"perk {old['perk'] or 'none'} → {x['perk'] or 'none'}")
        if ch:
            lines.append(f"team: {x['name']} " + ", ".join(ch))
    order_a = [x["id"] for x in a["team"] if x["id"] in tb]
    order_b = [x["id"] for x in b["team"] if x["id"] in ta]
    if order_a != order_b:
        lines.append("team order (front first): " + ", ".join(x["name"] for x in b["team"]))

    sb = {x["id"]: x for x in b["shop"]}
    if roll:
        lines.append("shop now: " + (", ".join(
            _item_str(x) + (" (frozen)" if x["frozen"] else "") for x in b["shop"]) or "empty"))
    else:
        for x in a["shop"]:
            if x["id"] not in sb and x["id"] not in tb:
                lines.append(f"shop: {_item_str(x)} {'used' if x['kind'] == 'food' else 'removed'}")
        for x in b["shop"]:
            if x["id"] not in sa:
                lines.append(f"shop: new {_item_str(x)}")
    for x in b["shop"]:
        old = sa.get(x["id"])
        if not old:
            continue
        if x["kind"] != "food" and (old["atk"], old["hp"]) != (x["atk"], x["hp"]):
            lines.append(f"shop: {x['name']} {old['atk']}/{old['hp']} → {x['atk']}/{x['hp']}")
        if old["frozen"] != x["frozen"]:
            lines.append(f"shop: {x['name']} {'frozen' if x['frozen'] else 'unfrozen'}")
    return lines


def violations(s):
    out = []
    if len(s["team"]) > 5:
        out.append(f"team has {len(s['team'])} pets, the limit is 5")
    if s["gold"] < 0:
        out.append(f"gold is negative ({s['gold']})")
    for x in s["team"]:
        if x["lvl"] > 3:
            out.append(f"{x['name']} is level {x['lvl']}, the max is 3")
        if x["hp"] <= 0:
            out.append(f"{x['name']} has {x['hp']} health in the shop phase")
    return out


def _ability(pet_name, level=1):
    e = data["pets"].get(pet_name, {})
    ab = e.get(f"level{level}Ability") or e.get("level1Ability") or {}
    return ab.get("description", "")


def _trigger(pet_name, level=1):
    e = data["pets"].get(pet_name, {})
    ab = e.get(f"level{level}Ability") or e.get("level1Ability") or {}
    return ab.get("trigger")


### Pet triggers that can fire during each kind of shop action
ACTION_TRIGGERS = {
    "buy_pet": ["Buy", "BuyTier1Animal", "Summoned"],
    "buy_combine": ["Buy", "BuyTier1Animal", "LevelUp"],
    "combine": ["LevelUp"],
    "buy_food": ["EatsShopFood", "PurchaseFood", "LevelUp"],
    "buy_food_untargeted": ["EatsShopFood", "PurchaseFood"],
    "sell": ["Sell"],
    "end_turn": ["EndOfTurn", "EndOfTurnAfterLoss", "EndOfTurnWithLvl3Friend"],
    "start_turn": ["StartOfTurn"],
}


def expectations(name, args, before):
    """(expected gold change, notes) for one env action from the basic rules."""
    shop = {x["i"]: x for x in before["shop"]}
    team = before["team"]
    notes = []
    gold = 0
    if name in ("buy_pet", "buy_combine", "buy_food", "buy_food_untargeted"):
        item = shop.get(args[0])
        gold = -(item["cost"] if item else 3)
        if name == "buy_combine" and len(args) > 1 and args[1] < len(team):
            t = team[args[1]]
            notes.append(f"rule: stacking keeps the higher stats +1/+1 and adds 1 experience to {t['name']}")
    elif name == "roll":
        gold = -1
    elif name == "sell":
        t = team[args[0]] if args[0] < len(team) else None
        gold = t["sell"] if t else 0
        if t:
            notes.append(f"rule: selling gives level + Cake bonus = {gold} gold")
    elif name == "combine":
        notes.append("rule: combining keeps the higher stats +1/+1 and adds 1 experience")
    return gold, notes


def _trigger_kind(pet_name, level=1):
    e = data["pets"].get(pet_name, {})
    ab = e.get(f"level{level}Ability") or e.get("level1Ability") or {}
    return (ab.get("triggeredBy") or {}).get("kind")


def acting_ids(name, args, before):
    """Ids of the pets an action is about: the pet sold, bought, fed or merged."""
    team, shop = before["team"], {x["i"]: x for x in before["shop"]}
    ids = set()
    if name in ("buy_pet", "buy_combine") and args and args[0] in shop:
        ids.add(shop[args[0]]["id"])
    if name in ("sell",) and args and args[0] < len(team):
        ids.add(team[args[0]]["id"])
    if name in ("buy_combine", "buy_food") and len(args) > 1 and args[1] < len(team):
        ids.add(team[args[1]]["id"])
    if name == "combine":
        ids.update(team[a]["id"] for a in args if a < len(team))
    return ids


def ability_notes(name, before, after, args=()):
    """Ability text of the pets this action can trigger, to explain extra changes.
    Abilities triggered by "Self" only count for the pet the action is about."""
    triggers = ACTION_TRIGGERS.get(name, [])
    if not triggers:
        return []
    acting = acting_ids(name, list(args), before)
    out, seen = [], set()
    for x in before["team"] + after["team"]:
        if x["id"] in seen:
            continue
        seen.add(x["id"])
        if _trigger(x["pet"], x["lvl"]) not in triggers:
            continue
        kind = _trigger_kind(x["pet"], x["lvl"])
        if kind == "Self" and name not in ("end_turn", "start_turn") and x["id"] not in acting:
            continue
        if kind == "EachFriend" and x["id"] in acting:
            continue
        out.append(f"ability in play: {x['name']} (L{x['lvl']}): {_ability(x['pet'], x['lvl'])}")
    return out


def gold_check(expected, before, after):
    actual = after["gold"] - before["gold"]
    sign = lambda v: ("+" if v > 0 else "−" if v < 0 else "±") + str(abs(v))
    if actual == expected:
        return f"check: gold {sign(actual)}, matches the rules", True
    return (f"check: gold {sign(actual)}, the basic rule says {sign(expected)}"
            " (the difference should come from an ability listed here)", False)
